# This file may not be shared/redistributed without permission. Please read copyright notice in the git repo. If this file contains other copyright notices disregard this text.
from unitgrade import UTestCase, Report
import irlc
import numpy as np

class SarlaccGameRules(UTestCase):
    def check_rules(self, rules):
        from irlc.project3i.sarlacc import game_rules
        # Test what happens at the starting square s=0 for roll 1
        self.assertEqualC(game_rules(rules, state=0, roll=1))
        # Test what happens at the starting square s=0 for other rolls
        for roll in [2, 3, 4, 5, 6]:
            self.assertEqualC(game_rules(rules, state=0, roll=roll))

        # Test all states:
        for s in range(max(rules.keys())):
            if s not in rules: # We skip because s is not a legal state to be in.
                for roll in [1, 2, 3, 4, 5, 6]:
                    self.assertEqualC(game_rules(rules, s, roll))

    def test_empty_board_rules(self):
        rules = {55: -1}
        self.check_rules(rules)

    def test_rules(self):
        from irlc.project3i.sarlacc import rules
        self.check_rules(rules)

class SarlacReturn(UTestCase):
    def check_return(self, rules, gamma):
        from irlc.project3i.sarlacc import sarlacc_return
        v = sarlacc_return(rules, gamma)
        # Check that the keys (states) that are included in v are correct. I.e., that the return is computed for the right states.
        states = list(sorted(v.keys()))
        self.assertEqualC(states)

        for s in states:
            self.assertL2(v[s], tol=1e-2)

    def test_sarlacc_return_empty_gamma1(self):
        self.check_return({55: -1}, gamma=1)

    def test_sarlacc_return(self):
        from irlc.project3i.sarlacc import rules
        self.check_return(rules, gamma=.8)


class Project3Individual(Report):
    title = "Project part 3: Reinforcement Learning (individual)"
    pack_imports = [irlc]

    sarlacc = [(SarlaccGameRules, 20),
               (SarlacReturn, 20)]

    questions = []
    questions += sarlacc


if __name__ == '__main__':
    from unitgrade import evaluate_report_student
    evaluate_report_student(Project3Individual())
