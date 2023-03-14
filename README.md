# icra23-code
Code for our ICRA 2023 Paper "Stochastic Planning for ASV Navigation using Satellite Images"

## Instructions for runnning PCCTP on toy graphs
1. Install dependencires `pip install -r requirements.txt`
2. `python src/test_graph.py`

Results (ao_tree, final policy, and video of the AO* search) will be saved under the `results` folder.

## Processed Dataset of Ontario Lake Water Masks
Coming soon!

## File Structure

- pcctp.py: Main module for solving pcctp
- test_graph.py: Toy Test Graph
- utils.py: Plotting utilites
- lake_graph.py: module for processing water masks into stochastic graphs