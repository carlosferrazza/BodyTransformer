CONDA_ENV_NAME=rlgpu_bot
eval "$(conda shell.bash hook)"
if conda info --envs | grep -q $CONDA_ENV_NAME; then
    echo "conda env $CONDA_ENV_NAME already exists."
    conda activate $CONDA_ENV_NAME
else
    echo "conda env $CONDA_ENV_NAME doesn't exist. Create a new one..."
    conda create -n $CONDA_ENV_NAME python=3.8.18 pip --yes
    conda activate $CONDA_ENV_NAME
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    echo "export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH" > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh

    conda install -y libgcc ninja

    pip install xvfbwrapper positional_encodings

    # Install isaacgym
    pip install -e /opt/isaacgym/python
    # Install IssacGymEnvs and rl_games. Assume we're in 'IssacGymEnvs' project's root directory.
    pip install -e .  # IssacGymEnvs
    pip install -e ./rl_games  # rl_games
fi
