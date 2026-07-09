"""Targeted regression tests for the bug fixes in utils / retrieve / IDesign.

These exercise the pure logic (no LLM, no GPU, no bpy):

1. get_depth now returns MINIMUM depth (BFS) instead of a depth that the old
   DFS could inflate via its `else` overwrite branch.
2. place_object / get_no_overlap_reason no longer share a mutable-default
   `errors` dict across calls.
3. retrieve.retrieve_candidates returns [] (not an IndexError) when no asset
   clears the similarity threshold.
4. IDesign.to_json creates parent directories.
5. JSONSchemaAgent (initial / corrector / refiner) constructs without autogen's
   Docker probe, accepts the iostream kwarg, and turns empty/non-JSON proxy
   replies into retry feedback instead of crashing.
"""

import importlib
import inspect
import json
import os


def _obj(oid, layout=None, room_objs=None):
    return {
        "new_object_id": oid,
        "placement": {
            "room_layout_elements": layout or [],
            "objects_in_room": room_objs or [],
        },
    }


def test_get_depth_returns_minimum_depth():
    utils = importlib.import_module("utils")
    # south_wall -> A(1) -> C(2) -> D(3)
    # south_wall -> B(1) -> D    (so D's MIN depth is 2, not 3)
    sg = [
        _obj("A", layout=[{"layout_element_id": "south_wall", "preposition": "on"}]),
        _obj("B", layout=[{"layout_element_id": "south_wall", "preposition": "on"}]),
        _obj("C", room_objs=[{"object_id": "A", "preposition": "on"}]),
        _obj("D", room_objs=[
            {"object_id": "B", "preposition": "on"},
            {"object_id": "C", "preposition": "on"},
        ]),
    ]
    d = utils.get_depth(sg)
    assert d == {"A": 1, "B": 1, "C": 2, "D": 2}, d


def test_place_object_does_not_share_errors_default():
    """Two independent place_object calls (omitting errors) must not accumulate
    state into a shared mutable default dict."""
    utils = importlib.import_module("utils")
    # An object absent from the scene_graph short-circuits with the empty
    # errors dict; this is enough to confirm the default is per-call.
    ghost = {"new_object_id": "does_not_exist"}
    e1 = utils.place_object(ghost, [], [4.0, 4.0, 2.5])
    e2 = utils.place_object(ghost, [], [4.0, 4.0, 2.5])
    assert e1 == {}
    assert e2 == {}
    # The default must be None (sentinel), not a shared dict instance.
    assert utils.place_object.__defaults__[0] is None


def test_get_no_overlap_reason_does_not_share_errors_default():
    utils = importlib.import_module("utils")
    obj = {
        "new_object_id": "x",
        "placement": {"room_layout_elements": [], "objects_in_room": []},
    }
    e1 = utils.get_no_overlap_reason(obj, [])
    e2 = utils.get_no_overlap_reason(obj, [])
    assert e1 == {} and e2 == {}
    assert utils.get_no_overlap_reason.__defaults__[1] is None


def test_retrieve_candidates_empty_without_backend():
    """retrieve_asset_for_object must return None (not raise IndexError, not
    trigger a multi-GB download) when the backend is not loaded. The old
    retrieve()[0] pattern crashed on an empty candidate list."""
    retrieve = importlib.import_module("retrieve")
    assert not retrieve.backend_available()
    obj = {"new_object_id": "chair_1", "style": "modern", "material": "wood"}
    # autoload=False (default) -> graceful None, no download.
    out = retrieve.retrieve_asset_for_object(
        obj, "/tmp/_retrieve_test_assets", match="best", sim_th=0.99, download=False
    )
    assert out is None
    # No embeddings bank should have been downloaded.
    assert not os.path.isdir("OpenShape-Embeddings")


def test_idesign_to_json_creates_parent_dirs(tmp_path):
    IDesign = importlib.import_module("IDesign")
    i = IDesign.IDesign.__new__(IDesign.IDesign)
    i.scene_graph = [{"new_object_id": "south_wall", "itemType": "wall"}]
    nested = tmp_path / "outputs" / "20260708-020000" / "scene_graph.json"
    i.to_json(str(nested))
    assert nested.exists()
    assert json.loads(nested.read_text()) == i.scene_graph


def _json_schema_agent_classes():
    """Return (label, class) for every JSONSchemaAgent subclass in the codebase.

    Finds them by scanning agent-defining modules so a future fourth copy can't
    sneak past this regression test.
    """
    agents = importlib.import_module("agents")
    corrector = importlib.import_module("corrector_agents")
    refiner = importlib.import_module("refiner_agents")
    found = []
    for mod in (agents, corrector, refiner):
        for name, obj in inspect.getmembers(mod, inspect.isclass):
            if name == "JSONSchemaAgent":
                found.append((mod.__name__, obj))
    assert found, "no JSONSchemaAgent classes discovered -- test harness is stale"
    return found


def test_json_schema_agents_skip_docker_probe():
    """Each JSONSchemaAgent must construct WITHOUT autogen probing Docker.

    The server's `docker` package is a stub (no from_env/errors attrs); a
    UserProxyAgent built with the default code_execution_config triggers the
    probe and crashes with AttributeError. code_execution_config=False skips it.
    This must hold for EVERY copy (initial/corrector/refiner).
    """
    agents = importlib.import_module("agents")
    is_term = agents.is_termination_msg
    for label, cls in _json_schema_agent_classes():
        # If the Docker probe fires, this raises AttributeError before we even
        # get to the assertions.
        inst = cls("Json_schema_debugger", is_term)
        # The base UserProxyAgent must not have armed code execution.
        assert inst._code_execution_config is False, (
            f"{label}: code_execution_config not disabled -> Docker probe would fire"
        )


def test_json_schema_agents_accept_iostream_kwarg():
    """autogen 0.14 calls get_human_input(prompt, iostream=...). The override
    must accept the iostream kwarg or it raises TypeError at runtime."""
    agents = importlib.import_module("agents")
    is_term = agents.is_termination_msg
    for label, cls in _json_schema_agent_classes():
        inst = cls("Json_schema_debugger", is_term)
        inst.last_message = lambda: {"name": "X", "content": ""}
        # Must not raise TypeError: unexpected keyword argument 'iostream'.
        out = inst.get_human_input("p", iostream=None)
        assert isinstance(out, str) and out, f"{label}: no feedback returned"


def test_json_schema_agents_tolerate_empty_and_invalid_json():
    """An empty / None / non-JSON reply from the proxy must become retry
    feedback (driving the debugger<->agent loop) rather than a
    JSONDecodeError: char 0 crash."""
    agents = importlib.import_module("agents")
    is_term = agents.is_termination_msg
    bad_contents = [
        "",                       # empty string
        None,                     # None content
        "the quick brown fox",    # garbage, no JSON
        "   ",                    # whitespace only
    ]
    for label, cls in _json_schema_agent_classes():
        for content in bad_contents:
            inst = cls("Json_schema_debugger", is_term)
            inst.last_message = lambda c=content: (
                {"name": "X", "content": c} if c is not None else {"name": "X", "content": None}
            )
            out = inst.get_human_input("p")
            assert out != "SUCCESS", f"{label}: empty/garbage content wrongly returned SUCCESS"
            assert "empty or not valid JSON" in out, (
                f"{label}: content={content!r} did not yield retry feedback (got: {out!r})"
            )


def test_json_schema_agents_return_success_on_valid_json():
    """A schema-valid reply must still return 'SUCCESS' (is_termination_msg
    keys off this)."""
    agents = importlib.import_module("agents")
    is_term = agents.is_termination_msg

    valid_init = {
        "objects_in_room": [{
            "new_object_id": "chair_1", "style": "modern", "material": "wood",
            "size_in_meters": {"length": 0.5, "width": 0.5, "height": 1.0},
            "is_on_the_floor": True, "facing": "north_wall",
            "placement": {
                "room_layout_elements": [{"layout_element_id": "south_wall", "preposition": "on"}],
                "objects_in_room": [],
            },
        }]
    }
    valid_corr = {"corrected_object": {
        "new_object_id": "chair_1", "is_on_the_floor": True, "facing": "north_wall",
        "placement": {
            "room_layout_elements": [{"layout_element_id": "south_wall", "preposition": "on"}],
            "objects_in_room": [],
        }}}
    valid_ref = {"children_objects": [{"name_id": "book_1", "placement": {
        "children_objects": [{"name_id": "book_2", "preposition": "left of", "is_adjacent": True}]}}]}
    valids = {"agents": valid_init, "corrector_agents": valid_corr, "refiner_agents": valid_ref}

    for label, cls in _json_schema_agent_classes():
        inst = cls("Json_schema_debugger", is_term)
        inst.last_message = lambda: {"name": "X", "content": json.dumps(valids[label])}
        assert inst.get_human_input("p") == "SUCCESS", (
            f"{label}: valid JSON did not return SUCCESS"
        )
