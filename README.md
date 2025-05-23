# OpenSCARA Simulation

This repository includes the code to simulate the SCARA in a PyBullet environment.

For a complete overview of the project, refer to the [main OpenSCARA repository](https://github.com/ggldnl/OpenSCARA).

## 🛠️ Build and deployment

1. Clone the repository:

   ```bash
    git clone --recursive https://github.com/ggldnl/OpenSCARA-Simulation.git
    ```

2. Create a conda environment:

    ```bash
    mamba env create -f environment.yml
    mamba activate openSCARA-sim
    ```

3. If for whatever reason you need to update the submodules:

    ```bash
    git submodule update --remote --recursive
    ```

## 🚀 Delpoy

- `simulation.py` will show the arm cycling through a list of points with straight lines.

    ```bash
    python simulation/showcase_animations.py -v ../media/rendering.mp4
    ```

    [![actions](https://img.youtube.com/vi/msuydRaIWuU/0.jpg)](https://www.youtube.com/watch?v=msuydRaIWuU)

## 🤝 Contribution

Feel free to contribute by opening issues or submitting pull requests. For further information, check out the [main OpenSCARA repository](https://github.com/ggldnl/OpenSCARA.git). Give a ⭐️ to this project if you liked the content.