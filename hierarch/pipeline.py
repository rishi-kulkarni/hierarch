from typing import Any, Callable, Dict, Generator, List, Tuple, Optional


class Pipeline:
    """Basic class for composing generators. The first generator added to
    the pipeline will be the "innermost" generator.

    Parameters
    ----------
    pipeline_components : Optional[List[Tuple[Callable, Dict]]], optional
        A pipeline can be initialized with a list of components, by default None
    """

    def __init__(
        self, pipeline_components: Optional[List[Tuple[Callable, Dict]]] = None
    ) -> None:
        if pipeline_components is not None:
            self._pipeline = pipeline_components
        else:
            self._pipeline = []

    @property
    def pipeline(self):
        return self._pipeline

    def add_component(self, component: Tuple[Callable, Dict]) -> None:
        """Add a component to the pipeline. The callable must be a
        generator function. The first argument to each component
        must be the result of the last component of the pipeline.

        Parameters
        ----------
        component : Tuple[Callable, Dict]
            Tuple of generator callable, keyword arguments to the callable.
        """
        self._pipeline.append(component)

    def process(self, data: Any) -> Generator[Any, None, None]:
        """Execute the pipeline to return a generator composed
        of all the generators in the pipeline.

        Parameters
        ----------
        data : Any
            The first argument to the first component of the pipeline.

        Returns
        -------
        Generator[Any, None, None]
            A generator that combines every generator in the pipeline.
        """
        generator = data
        for generator_function, kwargs in self._pipeline:
            generator = generator_function(generator, **kwargs)
        return generator
