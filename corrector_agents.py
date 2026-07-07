import autogen
from autogen.agentchat.groupchat import GroupChat
from autogen.agentchat.agent import Agent
from autogen.agentchat.user_proxy_agent import UserProxyAgent
from autogen.agentchat.assistant_agent import AssistantAgent
from copy import deepcopy
from jsonschema import validate
import json
import re

from schemas import layout_corrector_schema, deletion_schema
from agents import is_termination_msg

class JSONSchemaAgent(UserProxyAgent):
    def __init__(self, name : str, is_termination_msg):
        super().__init__(name, is_termination_msg=is_termination_msg)

    def get_human_input(self, prompt: str) -> str:
        message = self.last_message()
        preps_layout = ["left-side", "right-side", "in the middle"]
        preps_objs = ['on', 'left of', 'right of', 'in front', 'behind', 'under', 'above']

        content = message["content"]
        # The model may or may not wrap the JSON in ```json ... ``` fences;
        # accept either, falling back to the first {...} block, then raw content.
        pattern = r'```(?:json)?\s*(.+?)\s*```'
        m = re.search(pattern, content, re.DOTALL)
        if m is not None:
            match = m.group(1)
        else:
            brace = re.search(r'\{.*\}', content, re.DOTALL)
            match = brace.group(0) if brace is not None else content

        json_obj_new = json.loads(match)

        is_success  = False
        try:
            validate(instance=json_obj_new, schema=layout_corrector_schema)
            is_success = True
        except Exception as e:
            feedback = str(e.message)
            if e.validator == "enum":
                if str(preps_objs) in e.message:
                    feedback += f"Change the preposition {e.instance} to something suitable with the intended positioning from the list {preps_objs}"
                elif str(preps_layout) in e.message:
                    feedback += f"Change the preposition {e.instance} to something suitable with the intended positioning from the list {preps_layout}"
        if is_success:
            return "SUCCESS"
        return feedback

import os as _os
from typing import Optional
from agents import LLMConfig, build_configs

def get_corrector_agents(llm_config: Optional[LLMConfig] = None):
    cfg = build_configs(llm_config)
    gpt4_config = cfg["chat"]
    gpt4_json_config = cfg["json"]
    user_proxy = autogen.UserProxyAgent(
        name="Admin",
        system_message = "A human admin.",
        is_termination_msg = is_termination_msg,
        human_input_mode = "NEVER",
        code_execution_config=False
    )

    json_schema_debugger = JSONSchemaAgent(
        name = "Json_schema_debugger",
        is_termination_msg = is_termination_msg,
    )

    spatial_corrector_agent = AssistantAgent(
        name="Spatial_corrector_agent",
        llm_config=gpt4_config,
        is_termination_msg=is_termination_msg,
        human_input_mode="NEVER",
        system_message=f"""
        Spatial Corrector Agent. Whenever a user provides an object that don't fit the room for various spatial conflicts,
        You are going to make changes to its "scene_graph" and "facing_object" keys so that these conflicts are removed. 
        You are going to use the JSON Schema to validate the JSON object that the user provides.

        For relative placement with other objects in the room use the prepositions "on", "left of", "right of", "in front", "behind", "under".
        For relative placement with the room layout elements (walls, the middle of the room, ceiling) use the prepositions "on", "in the corner".

        Use only the following JSON Schema to save the JSON object:
        {layout_corrector_schema}
        """
    )

    object_deletion_agent = AssistantAgent(
        name="Object_deletion_agent",
        llm_config=gpt4_json_config,
        is_termination_msg=is_termination_msg,
        human_input_mode="NEVER",
        system_message=f"""
        Object Deletion Agent. When a user provides a list of objects that doesn't fit the room, select one object to delete that would be less essential for the room.

        An example JSON output:
        {deletion_schema}
        """
    )
    return user_proxy, json_schema_debugger, spatial_corrector_agent, object_deletion_agent