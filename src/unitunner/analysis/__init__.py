"""Inspect saved training artifacts without constructing model architectures."""


def analyze_run(project, *, output=None, show="auto", **options):
    """Analyze and save scientific results; automatically display in a notebook kernel."""
    from unitunner.analysis.run import analyze_run as execute

    return execute(project, output=output, show=show, **options)
