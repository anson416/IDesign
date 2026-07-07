import autogen
from autogen.agentchat.agent import Agent
from autogen.agentchat.user_proxy_agent import UserProxyAgent
from autogen.agentchat.assistant_agent import AssistantAgent
import json
import os
from dataclasses import dataclass, field
from jsonschema import validate
from copy import deepcopy
from typing import Optional

from schemas import initial_schema, interior_architect_schema, interior_designer_schema, engineer_schema


@dataclass
class LLMConfig:
    """Runtime-configurable LLM parameters threaded through every agent.

    All fields default to the original hardcoded/env behaviour so callers that
    pass nothing (e.g. upstream `test.py`) keep working unchanged.
    """

    model: str = "gpt-5.1-2025-11-13"
    base_url: str = field(
        default_factory=lambda: os.environ.get(
            "OPENAI_BASE_URL", "https://api.chatanywhere.tech/v1"
        )
    )
    api_key: str = field(
        default_factory=lambda: os.environ.get("OPENAI_API_KEY")
        or os.environ.get("CHATANYWHERE_API_KEY", "")
    )
    temperature: float = 0.7
    # Model used for JSON-object-mode calls (engineer + json schema debugger).
    # gpt-5.x + json_object triggers the chatanywhere proxy content filter, so
    # this defaults to gpt-4o unless overridden. None -> reuse `model`.
    json_model: Optional[str] = field(
        default_factory=lambda: os.environ.get(
            "VLMUNR_IDESIGN_JSON_MODEL", "gpt-4o"
        )
    )


def build_configs(llm_config: Optional[LLMConfig] = None) -> dict:
    """Build the four autogen llm_config dicts from an LLMConfig.

    Returns a dict with keys:
        manager   - chat config for GroupChatManager speaker selection
        chat      - assistant agents using plain text (designer, architect,
                    spatial corrector)
        json      - assistant agents using response_format json_object
        engineer  - engineer agent (json_object, temperature from config)

    The temperature is applied uniformly to all agents (per the CLI
    requirement); previously the engineer was pinned to 0.0.
    """
    cfg = llm_config if llm_config is not None else LLMConfig()
    model = cfg.model
    json_model = cfg.json_model if cfg.json_model else model

    chat_entry = {"model": model, "api_key": cfg.api_key, "base_url": cfg.base_url}
    json_entry = {
        "model": json_model,
        "api_key": cfg.api_key,
        "base_url": cfg.base_url,
    }

    base = {
        "cache_seed": 42,
        "temperature": cfg.temperature,
        "config_list": [chat_entry],
        "timeout": 600,
    }
    chat_config = deepcopy(base)

    json_config = deepcopy(base)
    json_config["config_list"] = [
        dict(json_entry, **{"response_format": {"type": "json_object"}})
    ]

    engineer_config = deepcopy(base)
    engineer_config["temperature"] = cfg.temperature
    engineer_config["config_list"] = [
        dict(json_entry, **{"response_format": {"type": "json_object"}})
    ]

    return {
        "manager": chat_config,
        "chat": chat_config,
        "json": json_config,
        "engineer": engineer_config,
    }


# ---- Backward-compatible module-level configs (used by upstream test.py). ----
_DEFAULT_CFG = build_configs()
gpt4_prev_config = _DEFAULT_CFG["chat"]
gpt4_config = _DEFAULT_CFG["manager"]
gpt4_json_config = _DEFAULT_CFG["json"]
gpt4_json_engineer_config = _DEFAULT_CFG["engineer"]

def is_termination_msg(content) -> bool:
    have_content = content.get("content", None) is not None
    if have_content and content["name"] == "Json_schema_debugger" and "SUCCESS" in content["content"]:
        return True
    return False


class JSONSchemaAgent(UserProxyAgent):
    def __init__(self, name : str, is_termination_msg):
        super().__init__(name, is_termination_msg=is_termination_msg)

    def get_human_input(self, prompt: str) -> str:
        message = self.last_message()
        preps_layout = ['in front', 'on', 'in the corner', 'in the middle of']
        preps_objs = ['on', 'left of', 'right of', 'in front', 'behind', 'under', 'above']

        json_obj_new = json.loads(message["content"])
        try:
            json_obj_new_ids = [item["new_object_id"] for item in json_obj_new["objects_in_room"]]
        except:
            return "Use 'new_object_id' instead of 'object_id'!"

        is_success  = False
        try:
            validate(instance=json_obj_new, schema=initial_schema)
            is_success = True
        except Exception as e:
            feedback = str(e.message)
            if e.validator == "enum":
                if e.instance in json_obj_new_ids:
                    feedback += f" Put the {e.instance} object under 'objects_in_room' instead of 'room_layout_elements' and delete the {e.instance} object under 'room_layout_elements'"
                elif str(preps_objs) in e.message:
                    feedback += f"Change the preposition {e.instance} to something suitable with the intended positioning from the list {preps_objs}"
                elif str(preps_objs) in e.message:
                    feedback += f"Change the preposition {e.instance} to something suitable with the intended positioning from the list {preps_layout}"

        if is_success:
            return "SUCCESS"
        return feedback

def create_agents(no_of_objects : int, llm_config: Optional[LLMConfig] = None):
    cfg = build_configs(llm_config)
    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        system_message = "A human admin.",
        is_termination_msg = is_termination_msg,
        code_execution_config=False
    )

    json_schema_debugger = JSONSchemaAgent(
        name = "Json_schema_debugger",
        is_termination_msg = is_termination_msg,
    )
    interior_designer = autogen.AssistantAgent(
        name = "Interior_designer",
        llm_config = cfg["json"],
        human_input_mode = "NEVER",
        is_termination_msg = is_termination_msg,
        system_message = f""" Interior Designer. Suggest {no_of_objects} essential new objects to be added to the room based on the user preference, general functionality of the room and the room size.
        The suggested objects should contain the following information:

        1. Object name (ex. bed, desk, chair, monitor, bookshelf, etc.)
        2. Architecture style (ex. modern, classic, etc.)
        3. Material (ex. wood, metal, etc.)
        4. Bounding box size in meters (ex. Length : 1.0m, Width : 1.0m, Height : 1.0m). Only use "Length", "Width", "Height" as keys for the size of the bounding box!
        5. Quantity (ex. 1, 2, 3, etc.)

        IMPORTANT: Do not suggest any objects related to doors or windows, such as curtains, blinds, etc.

        Follow the JSON schema below:
        {interior_designer_schema}

        """
    )


    interior_architect = autogen.AssistantAgent(
        name = "Interior_architect",
        llm_config = cfg["json"],
        human_input_mode = "NEVER",
        is_termination_msg = is_termination_msg,
        system_message = f""" Interior Architect. Your role is to analyze the user preference, think about where the optimal
        placement for each object would be that the Interior Designer suggests and find a place for this object in the room and give a detailed description of it.
        If the quantity of an object is greater than one, you have to find a place for each instance of this object separately!, but give all this information in one list item!
        Give explicit answers for EACH object on the following three aspects:

        Placement: 
        Find a relative place for the object (ex. on the middle of the floor, in the north-west corner, on the east wall, right of the desk, on the bookshelf...).
        For relative placement with other objects in the room use the prepositions "on", "left of", "right of", "in front", "behind", "under".
        For relative placement with the room layout elements (walls, the middle of the room, ceiling) use the prepositions "on", "in the corner".
        You are not allowed to use any prepositions different from the ones above!! 
        Expliticly state the placement for each instance (ex. one is left of desk_1, one is on the south_wall)!!

        Proximity : 
        Proximity of this object to the relative placement objects:
        1. Adjacent : The object is physically contacting the other object or it is supported by the other object or they are touching or they are close to each other.
        2. Not Adjacent: The object is not physically contacting the other object and it is distant from the other object.


        Facing :
        Think about which wall (west/east/north/south_wall) this object should be facing and explicitly state this (ex. one is facing the south_wall, one is facing the west_wall)!

        Follow the JSON schema below:
        {interior_architect_schema}

        If the quantity of an object is greater than one, you have to find a place for each instance of this object separately, but give all this information in one list item!
        This means the output should have one dictionary for each object, but all their instances (quantity higher than one) should be in the same dictionary!

        JSON
        """
    )

    engineer = autogen.AssistantAgent(
        name = "Engineer",
        llm_config = cfg["engineer"],
        human_input_mode = "NEVER",
        is_termination_msg = is_termination_msg,
        system_message = f""" Engineer. You listen to the input by the Admin and create a JSON file.
        Every time when the Admin outputs objects to be in the room you will save ALL of them in the given schema!
        For the scene graph, you can use the ids for the objects that are already in the room, but only output the objects to be placed!
        If an object has a quantity higher than one, save each instance of this object separately!!
        If the Json_schema_debugger reports a validation error about the JSON schema, solve the error in a way that spatially makes sense!

        IMPORTANT: The inputted "Placement" key should be used for the "placement" key in the JSON object follow exatly the prepositions stated, 
        do not use the information in "Facing" key for the room layout elements!!!

        IMPORTANT: For object quantities greater than one, the "placement" key gives separately the relative placement of each instance of that object in the room
        make the distinction accordingly!

        Use only the following JSON Schema to save the JSON object:
        {engineer_schema}

        """
    )

    return user_proxy, json_schema_debugger, interior_designer, interior_architect, engineer

