import subprocess as sp

from ..exceptions import ProgramNotFoundError

ATTRIBUTES = {
    "graph": {"rankdir": "TD", "bgcolor": "transparent", "nodesep": 1, "ranksep": 1},
    "node": {
        "shape": "circle",
        "label": "",
        "width": 0.2,
        "height": 0.2,
        "fontsize": 12,
    },
    "edge": {
        "dir": "both",
        "arrowhead": "none",
        "arrowtail": "none",
        "arrowsize": 0.2,
        "fontsize": 8,
    },
}

STYLES = {
    "root": {
        "shape": "circle",
        "color": "#a0a0a0",
        "style": "filled",
        "width": 0.05,
        "height": 0.05,
    },
    "node": {"color": "#a0a0a0"},
    "node_start": {"fillcolor": "#a0f0a0", "style": "filled"},
    "node_end": {"fillcolor": "#a0a0f0", "style": "filled"},
    "node_keep": {"fillcolor": "#a0f0a0", "style": "filled"},
    "nodes_path": {"fillcolor": "#a0a0a0", "style": "filled"},
    "node_squeeze": {"fillcolor": "#f0a0a0", "shape": "house", "style": "filled"},
    "node_expand": {"fillcolor": "#a0f0a0", "shape": "invhouse", "style": "filled"},
    "node_dim": {"shape": "box"},
    "edge": {"color": "#a0a0a0"},
    "edge_root": {"style": "dotted", "color": "#a0a0a0"},
    "edge_down": {"arrowhead": "normal", "style": "bold"},
    "edge_up": {"arrowtail": "normal", "style": "bold"},
}


def get_dot_cmd(filetype="png", dpi=300):
    """Return command for subprocess"""
    return ["dot", "-T%s" % filetype, "-Gdpi=%d" % dpi]


def get_image_bytes(dot_cmd, dot_str):
    """return image as bytes from running dot in subprocess"""
    try:
        proc = sp.Popen(dot_cmd, stdin=sp.PIPE, stdout=sp.PIPE, stderr=sp.PIPE)
    except FileNotFoundError:
        raise ProgramNotFoundError(dot_cmd[0])
    out_img, err = proc.communicate(input=dot_str.encode())
    err_msg = err.decode()
    if err_msg:
        raise Exception(err_msg)
    return out_img


def get_val_str(val):
    """create value string for dot"""
    if val is None:
        return "none"
    if isinstance(val, (list, tuple)):
        return ",".join(str(x) for x in val)
    return val


def get_attribute_str(attributes):
    """create attribute string for dot"""
    return ",".join('%s="%s"' % (k, get_val_str(v)) for k, v in attributes.items())


def get_node_str(node_id, attributes):
    """create node string for dot"""
    return "%s[%s];" % (node_id, get_attribute_str(attributes))


def get_edge_str(node_ids, attributes):
    """create edge string for dot"""
    return "%s -> %s [%s];" % (node_ids[0], node_ids[1], get_attribute_str(attributes))


def get_dot_digraph_str(dot_components):
    """create dot string from parts"""
    components_str = []
    for comp in dot_components:
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
    """create image from variable transormation steps
    Args:
        variable: Variable object
        filetype(str): "png" or "svg"
        dpi(int): resolution for png image
    """
    steps = variable.get_transform_steps(variable._domain)
    return draw_transform(steps, filetype=filetype, dpi=dpi)


def draw_transform(dim_steps, filetype="png", dpi=150, weight_names=None):
    """create image from variable transormation steps
    Args:
        dim_steps(OrderedDict): dimension -> steps
          * each element contains steps for a dimension
          * dimensions are all dimensions in source and target domain
          * each step is (from_level, to_level, action, (weight_level, weight_var))
        filetype(str, optional): "png" or "svg"
        dpi(int, optional): resolution for png image, defaults to 150
        weight_names(dict, optional): str -> str mapping of weight level
          to variable name that will be shown on edges
    """
    dot_cmd = get_dot_cmd(filetype=filetype, dpi=dpi)
    dot_components = get_components(dim_steps, weight_names=weight_names)
    dot_str = get_dot_digraph_str(dot_components)
    image_bytes = get_image_bytes(dot_cmd, dot_str)

    return image_bytes


def get_transform_path(steps, weight_names=None):
    """identify parts of subgraph that are part of the transformation path"""

    transform_path = {
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
                transform_path["nodes_path"].add(from_level)
                if not transform_path["node_start"]:
                    transform_path["node_start"] = from_level
            if to_level:
                transform_path["nodes_path"].add(to_level)
                transform_path["node_end"] = to_level
        if action == "aggregate":
            transform_path["edges_up"].add((from_level, to_level))
        elif action == "disaggregate":
            transform_path["edges_down"].add((from_level, to_level))
        elif action == "squeeze":
            transform_path["squeeze"] = True
        elif action == "expand":
            transform_path["expand"] = True
        elif action == "keep":
            # from_level == to_level
            transform_path["node_keep"] = from_level
        else:
            raise NotImplementedError(action)

        if weight:
            # determine weight name
            weight_name = "<weight>"
            if weight_names:
                if action == "aggregate" and from_level.name in weight_names:
                    weight_name = weight_names[from_level.name]
                elif action == "disaggregate" and to_level.name in weight_names:
                    weight_name = weight_names[to_level.name]
            key = (from_level, to_level)
            transform_path["edges_weight"][key] = weight_name

    if transform_path["node_start"]:
        transform_path["nodes_path"].remove(transform_path["node_start"])

    if transform_path["node_end"]:
        transform_path["nodes_path"].remove(transform_path["node_end"])

    return transform_path


def iter_tree(node, parent=None):
    yield (node, parent)
    for child in node.children.values():
        yield from iter_tree(child, parent=node)


def get_components(dim_steps, weight_names=None):
    # create node ids for for dot
    node_ids = {}

    def get_node_id(*args):
        if args not in node_ids:
            node_ids[args] = "N%d" % len(node_ids)
        return node_ids[args]

    dot_components = [
        # global config
        ("graph", ATTRIBUTES["graph"]),
        ("node", ATTRIBUTES["node"]),
        ("edge", ATTRIBUTES["edge"]),
        # root node
        (get_node_id(None), STYLES["root"]),
    ]

    # iterate over dimensions and create subgraphs
    for dim_idx, (dim, steps) in enumerate(dim_steps.items()):
        transform_path = get_transform_path(steps, weight_names=weight_names)

        # start cluster for dimension
        dot_components.append("subgraph cluster_%d {" % dim_idx)
        dot_components.append("peripheries=0;")  # no border around cluster

        # generate all nodes for this dimension
        for node, parent in iter_tree(dim):
            nid = get_node_id(node)
            # add node
            node_attrs = get_node_attrs(node, parent, transform_path)
            dot_components.append((nid, node_attrs))

        # close cluster for dimension
        dot_components.append("}")

        # generate all nodes for this dimension
        for node, parent in iter_tree(dim):
            nid = get_node_id(node)
            edge_down_id = (get_node_id(parent), nid)
            edge_attrs = get_edge_attrs(node, parent, transform_path)
            dot_components.append(
                (
                    edge_down_id,
                    edge_attrs,
                )
            )

    return dot_components


def get_node_attrs(node, parent, transform_path):
    node_attr = STYLES["node"].copy()
    node_attr.update({"xlabel": node.name})
    if node == transform_path["node_start"]:
        node_attr.update(STYLES["node_start"])
    elif node == transform_path["node_end"]:
        node_attr.update(STYLES["node_end"])
    elif node == transform_path["node_keep"]:
        node_attr.update(STYLES["node_keep"])
    elif node in transform_path["nodes_path"]:
        node_attr.update(STYLES["nodes_path"])
    if not parent:
        if transform_path["squeeze"]:
            node_attr.update(STYLES["node_squeeze"])
        elif transform_path["expand"]:
            node_attr.update(STYLES["node_expand"])
        else:
            node_attr.update(STYLES["node_dim"])
    return node_attr


def get_edge_attrs(node, parent, transform_path):
    edge_attr = STYLES["edge"].copy()
    if not parent:  # edge from root node
        edge_key_down = (None, node)
        edge_attr.update(STYLES["edge_root"])
    else:
        edge_key_down = (parent, node)
    edge_key_up = tuple(reversed(edge_key_down))
    if (
        edge_key_down in transform_path["edges_down"]
        or edge_key_up in transform_path["edges_down"]
    ):
        edge_attr.update(STYLES["edge_down"])
    elif (
        edge_key_down in transform_path["edges_up"]
        or edge_key_up in transform_path["edges_up"]
    ):
        edge_attr.update(STYLES["edge_up"])
    weight_name = transform_path["edges_weight"].get(edge_key_down) or transform_path[
        "edges_weight"
    ].get(edge_key_up)
    if weight_name:
        edge_attr.update({"xlabel": weight_name})
    return edge_attr
