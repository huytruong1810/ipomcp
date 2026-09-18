"""Resolve figure labels from explicit inputs and recorded data."""


def agent_labels(df, explicit=None):
    if explicit is not None:
        if set(explicit) != {"i", "j"}:
            raise ValueError("Agent labels must contain exactly i and j")
        return dict(explicit)
    labels = {}
    for agent in ("i", "j"):
        column = f"level_{agent}"
        label = f"Agent {agent.upper()}"
        if column in df:
            values = df[column].dropna().unique()
            if len(values) != 1 or values[0] != int(values[0]):
                raise ValueError(f"{column} must contain one integer level per figure")
            label += f" (Level-{int(values[0])})"
        labels[agent] = label
    return labels
