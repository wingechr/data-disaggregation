import logging
import subprocess as sp
from tempfile import NamedTemporaryFile

from ..exceptions import ProgramNotFoundError


def get_dot_cmd(filetype="png", dpi=300, **kwargs):
    # use neato -n for pre-calculated positions
    return ["dot", "-T%s" % filetype, "-Gdpi=%d" % dpi]


def get_image_bytes(dot_cmd, dot_str):
    logging.debug(" ".join(dot_cmd))
    try:
        proc = sp.Popen(dot_cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
    except FileNotFoundError:
        raise ProgramNotFoundError(dot_cmd[0])
    out_img, err = proc.communicate(input=dot_str.encode())
    err_msg = err.decode()
    if err_msg:
        raise Exception(err_msg)
    return out_img


def notebook_display_image_bytes(image_bytes, filetype="png"):
    from IPython import display

    with NamedTemporaryFile(suffix="." + filetype, delete=False, mode="wb") as file:
        file.write(image_bytes)
    return display.Image(file.name)


def get_val_str(val):
    if val is None:
        return "none"
    if isinstance(val, (list, tuple)):
        return ",".join(str(x) for x in val)
    return val


def get_attribute_str(attributes):
    return ",".join('%s="%s"' % (k, get_val_str(v)) for k, v in attributes.items())


def get_node_str(node_id, attributes):
    return "%s[%s];" % (node_id, get_attribute_str(attributes))


def get_edge_str(node_ids, attributes):
    return "%s -> %s [%s];" % (node_ids[0], node_ids[1], get_attribute_str(attributes))


def get_dot_str(components):
    components_str = []

    for comp in components:
        if isinstance(comp, str):
            cmp_str = comp
        else:
            node_id, attributes = comp
            if isinstance(node_id, tuple):
                cmp_str = get_edge_str(node_id, attributes)
            else:
                cmp_str = get_node_str(node_id, attributes)
        components_str.append(cmp_str)
    components_str = "\n".join(components_str)
    return "digraph{\n%s\n}" % components_str


def draw_domain(variable, filetype="png", dpi=150):
    steps = variable.get_transform_steps(variable._domain)
    return draw_transform(steps, filetype=filetype, dpi=dpi)


def draw_transform(dim_steps, filetype="png", dpi=150):
    """
    Args:
        dim_steps(OrderedDict): dimension -> steps
          * each element contains steps for a dimension
          * dimensions are all dimensions in source and target domain
          * each step is (from_level, to_level, action, (weight_level, weight_var))
    """
    components = [
        # global config
        ("graph", {"rankdir": "TD", "nodesep": 1, "ranksep": 1}),
        (
            "node",
            {
                "shape": "circle",
                "label": "",
                "width": 0.2,
                "height": 0.2,
                "fontsize": 12,
            },
        ),
        (
            "edge",
            {
                "dir": "both",
                "arrowhead": "none",
                "arrowtail": "none",
                "arrowsize": 0.2,
                "fontsize": 8,
            },
        ),
    ]

    node_ids = {}

    def get_id(*args):
        if args not in node_ids:
            node_ids[args] = "N%d" % len(node_ids)
        return node_ids[args]

    components.append(
        (
            get_id(None),
            {
                "shape": "circle",
                "color": "#a0a0a0",
                "style": "filled",
                "width": 0.05,
                "height": 0.05,
            },
        )
    )  # root node

    def add_rec(dim_lev, transf, components, parent=None):
        nid = get_id(dim_lev)
        pid = get_id(parent)
        ids_down = (pid, nid)

        edge_attr = {"color": "#a0a0a0"}
        node_attr = {"color": "#a0a0a0", "xlabel": dim_lev.name}

        if dim_lev == transf["node_start"]:
            node_attr.update({"fillcolor": "#a0f0a0", "style": "filled"})
        elif dim_lev == transf["node_end"]:
            node_attr.update({"fillcolor": "#a0a0f0", "style": "filled"})
        elif dim_lev == transf["node_keep"]:
            node_attr.update({"fillcolor": "#a0f0a0", "style": "filled"})
        elif dim_lev in transf["nodes_path"]:
            node_attr.update({"fillcolor": "#a0a0a0", "style": "filled"})

        else:
            pass

        if not parent:  # dim level
            edge_key_down = (None, dim_lev)
            if transf["squeeze"]:
                node_attr.update(
                    {"fillcolor": "#f0a0a0", "style": "filled", "shape": "house"}
                )
            elif transf["expand"]:
                node_attr.update(
                    {"fillcolor": "#a0f0a0", "style": "filled", "shape": "invhouse"}
                )
            else:
                node_attr.update({"shape": "box"})

        else:
            edge_key_down = (parent, dim_lev)

        edge_key_up = tuple(reversed(edge_key_down))

        if edge_key_down in transf["edges_down"] or edge_key_up in transf["edges_down"]:
            edge_attr.update({"arrowhead": "normal", "style": "bold"})
        elif edge_key_down in transf["edges_up"] or edge_key_up in transf["edges_up"]:
            edge_attr.update({"arrowtail": "normal", "style": "bold"})

        weight = transf["edges_weight"].get(edge_key_down) or transf[
            "edges_weight"
        ].get(edge_key_up)
        if weight:
            edge_attr.update({"xlabel": weight})

        components.append((nid, node_attr))
        if parent:
            # add root edges later (outside of cluster)
            components.append((ids_down, edge_attr))

        # recursion
        for child in dim_lev.children.values():
            add_rec(child, transf, components, parent=dim_lev)

    for dim_idx, (dim, steps) in enumerate(dim_steps.items()):

        transf = {
            "edges_up": set(),
            "edges_down": set(),
            "squeeze": False,
            "expand": False,
            "node_start": None,
            "node_end": None,
            "node_keep": None,
            "nodes_path": set(),
            "edges_weight": {},
        }

        for from_level, to_level, action, weight in steps:
            if action != "keep":
                if from_level:
                    transf["nodes_path"].add(from_level)
                    if not transf["node_start"]:
                        transf["node_start"] = from_level
                if to_level:
                    transf["nodes_path"].add(to_level)
                    transf["node_end"] = to_level

            if action == "aggregate":
                transf["edges_up"].add((from_level, to_level))
            elif action == "disaggregate":
                transf["edges_down"].add((from_level, to_level))
            elif action == "squeeze":
                transf["squeeze"] = True
            elif action == "expand":
                transf["expand"] = True
            elif action == "keep":
                # from_level == to_level
                transf["node_keep"] = from_level
            else:
                raise NotImplementedError(action)

            if weight:
                key = (from_level, to_level)
                transf["edges_weight"][key] = weight

        if transf["node_start"]:
            transf["nodes_path"].remove(transf["node_start"])

        if transf["node_end"]:
            transf["nodes_path"].remove(transf["node_end"])

        dim_components = []
        add_rec(dim, transf, dim_components)

        components.append("subgraph cluster_%d {" % dim_idx)
        # cluster styles
        components.append("peripheries=0;")  # no border around cluster
        components += dim_components
        components.append("}")

    for dim in dim_steps.keys():
        # root edge
        edge_attr = {
            "style": "dotted",
            "color": "#a0a0a0",
        }  # invisible edge from root node
        nid = get_id(dim)
        pid = get_id(None)
        ids_down = (pid, nid)
        components.append((ids_down, edge_attr))

    dot_cmd = get_dot_cmd(filetype=filetype, dpi=dpi)
    dot_str = get_dot_str(components)
    image_bytes = get_image_bytes(dot_cmd, dot_str)
    return image_bytes
