import unittest
import pandas as pd
from recommender import get_recommendations

class TestNetflixLogic(unittest.TestCase):
    def setUp(self):
        # Small mock dataset for testing
        self.mock_df = pd.DataFrame({
            'title': ['Movie A', 'Movie B', 'Movie C'],
            'description': ['A story about space and stars', 'Astronauts in space', 'A romantic dinner'],
            'listed_in': ['Sci-Fi', 'Sci-Fi', 'Romance'],
            'cast': ['Actor 1', 'Actor 1', 'Actor 2']
        })

    def test_recommender(self):
        result = get_recommendations(self.mock_df, 'Movie A')
        # Movie B should be more similar to A than Movie C is
        self.assertIn('Movie B', result)

if __name__ == '__main__':
    unittest.main()
    import unittest
import pandas as pd
from recommender import get_recommendations

class TestNetflixLogic(unittest.TestCase):
    def setUp(self):
        # Small mock dataset for testing
        self.mock_df = pd.DataFrame({
            'title': ['Movie A', 'Movie B', 'Movie C'],
            'description': ['A story about space and stars', 'Astronauts in space', 'A romantic dinner'],
            'listed_in': ['Sci-Fi', 'Sci-Fi', 'Romance'],
            'cast': ['Actor 1', 'Actor 1', 'Actor 2']
        })

    def test_recommender(self):
        result = get_recommendations(self.mock_df, 'Movie A')
        # Movie B should be more similar to A than Movie C is
        self.assertIn('Movie B', result)

if __name__ == '__main__':
    unittest.main()
