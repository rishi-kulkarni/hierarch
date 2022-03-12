from unittest import TestCase
from hierarch.pipeline import Pipeline


class TestPipeline(TestCase):
    def test_initialized_pipeline(self):

        pipeline = Pipeline([(range, {}), (lambda x, y: (a * y for a in x), {"y": 3})])

        expected = [0, 3, 6, 9]
        generated = list(pipeline.process(4))

        self.assertEqual(generated, expected)

    def test_built_pipeline(self):

        pipeline = Pipeline()

        pipeline.add_component(range)
        pipeline.add_component(lambda x: (a * 10 for a in x))

        expected = [0, 10, 20, 30]
        generated = list(pipeline.process(4))

        self.assertEqual(generated, expected)
