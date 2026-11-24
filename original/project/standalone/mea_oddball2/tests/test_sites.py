import json
import math
from pathlib import Path
import unittest

from oddball2 import sites

REAL = (Path(__file__).resolve().parents[3] / 'output' / 'mea_oddball_22021_20261123' / 'records'
        / 'oddball22021-20261123-site-selection' / 'site-candidates.json')


def row(electrode, activity=5):
    x, y = sites.electrode_xy(electrode)
    return dict(channel=electrode % 1024, electrode=electrode, min_spikes_per_20s=activity, x_um=x, y_um=y)


def patch(row0, column0, size=24, step=3):
    """Electrodes of a square patch of active tissue: 64 electrodes, about 370 um wide."""
    return [(row0 + r) * sites.COLUMNS + column0 + c for r in range(0, size, step) for c in range(0, size, step)]


class GeometryTests(unittest.TestCase):
    def test_electrode_positions_match_the_recorded_route_of_chip_22021(self):
        self.assertEqual(sites.electrode_xy(73), (1277.5, 0.))
        self.assertEqual(sites.electrode_xy(10562), (35., 840.))
        self.assertEqual(sites.electrode_xy(11590), (2625., 910.))
        for bad in (-1, 26400, 1.5, '7'):
            with self.assertRaises(ValueError):
                sites.electrode_xy(bad)

    def test_neighbours_are_the_electrodes_within_40_um_nearest_first(self):
        centre = 60 * sites.COLUMNS + 100
        found = sites.neighbours(centre)
        self.assertEqual(len(found), 20)
        self.assertNotIn(centre, found)
        cx, cy = sites.electrode_xy(centre)
        distances = [math.hypot(x - cx, y - cy) for x, y in map(sites.electrode_xy, found)]
        self.assertEqual(distances, sorted(distances))
        self.assertEqual(distances[:4], [17.5] * 4)
        self.assertLessEqual(max(distances), 40.)
        self.assertEqual(set(found[:4]), {centre - 1, centre + 1, centre - sites.COLUMNS, centre + sites.COLUMNS})
        corner = sites.neighbours(0)
        self.assertEqual(len(corner), 7)
        self.assertTrue(all(0 <= e < sites.COLUMNS * sites.ROWS for e in corner))
        edge = sites.neighbours(219)                       # last column of the first row: no wrap to the next row
        self.assertTrue(all(sites.electrode_xy(e)[0] >= 3780. for e in edge))


class ChoiceTests(unittest.TestCase):
    def setUp(self):
        # Three patches of active tissue 700 um apart, plus two very active electrodes alone at the array edge.
        self.tissue = patch(20, 40) + patch(20, 80) + patch(60, 60)
        self.lonely = [5, 119 * sites.COLUMNS + 215]
        self.rows = [row(e) for e in self.tissue] + [row(e, 500) for e in self.lonely]
        self.active = [sites.electrode_xy(e) for e in self.tissue + self.lonely]

    def test_sites_lie_in_active_tissue_and_not_at_the_most_active_lonely_electrode(self):
        chosen = sites.choose(self.rows, self.active)
        picked = [chosen['sites'][s]['electrode'] for s in 'HWF']
        self.assertFalse(set(picked) & set(self.lonely))
        self.assertTrue(all(chosen['sites'][s]['near_active'] >= sites.MIN_NEAR for s in 'HWF'))
        self.assertTrue(all(d >= sites.MIN_SEPARATION_UM for d in chosen['distances_um'].values()))
        self.assertEqual(sites.choose(self.rows, self.active), sites.choose(self.rows[::-1], self.active))

    def test_tissue_that_cannot_hold_three_separated_sites_is_refused(self):
        one = patch(20, 40)                                # a single patch: no three points 500 um apart
        with self.assertRaises(ValueError):
            sites.choose([row(e) for e in one], [sites.electrode_xy(e) for e in one])
        sparse = self.tissue[::10]                         # active but too sparse: fewer than 20 channels nearby
        with self.assertRaises(ValueError):
            sites.choose([row(e) for e in sparse], [sites.electrode_xy(e) for e in sparse])

    def test_surroundings_count_the_annulus_and_skip_channels_next_to_any_centre(self):
        centres = dict(H=(1000., 1000.), W=(1700., 1000.), F=(1000., 1700.))
        active = {1: (1050., 1000.), 2: (1150., 1000.), 3: (1500., 1000.), 4: (1650., 1000.), 5: (1000., 1501.)}
        near = sites.surroundings(centres, active)
        self.assertEqual(near['H'], [2, 3])                 # 1 is 50 um from H; 4 is next to W; 5 is 501 um away
        self.assertEqual(near['W'], [3])
        self.assertEqual(near['F'], [5])

    @unittest.skipUnless(REAL.is_file(), 'recorded candidates of chip 22021 not present')
    def test_chip_22021_gets_three_sites_inside_its_active_region(self):
        rows = json.loads(REAL.read_text())['candidates']
        chosen = sites.choose(rows, [(r['x_um'], r['y_um']) for r in rows])
        for site in 'HWF':
            self.assertGreaterEqual(chosen['sites'][site]['near_active'], sites.MIN_NEAR)
        self.assertGreaterEqual(min(chosen['distances_um'].values()), sites.MIN_SEPARATION_UM)


if __name__ == '__main__':
    unittest.main()
