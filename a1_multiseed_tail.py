"""Tail runner for :mod:`a1_multiseed`, only the configs that the
main job did not finish before its SLURM timeout: rel-stack, seed=2024,
directions D and DirGNN. Two configs, ~30 min total on an L40.
"""
import a1_multiseed as m

m.DATASETS = [("rel-stack", "user-engagement")]
m.SEEDS = [2024]
m.DIRECTIONS = ["D", "DirGNN"]

if __name__ == "__main__":
    m.main()
