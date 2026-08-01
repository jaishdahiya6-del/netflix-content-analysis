import unittest
import pandas as pd
import sys
import os

# Add src to python path so we can import from it easily
base_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(base_dir, 'src')
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

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
