#!/bin/bash

# 获取 Conda 环境目录
CONDA_ENVS_DIR="/datadisk4/cs/anaconda3/envs"

# 遍历所有 Conda 环境
for env_dir in "$CONDA_ENVS_DIR"/*; do
    if [ -d "$env_dir" ]; then
        env_name=$(basename "$env_dir")
        pip_script="$env_dir/bin/pip"

        if [ -f "$pip_script" ]; then
            echo "Fixing pip path for environment: $env_name"
            sed -i "1s|.*|#!/datadisk4/cs/anaconda3/envs/$env_name/bin/python|" "$pip_script"
        else
            echo "No pip script found for environment: $env_name"
        fi
    fi
done
