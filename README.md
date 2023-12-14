# TNL2K_Evaluation_Toolkit_Python

## Tutorial for the Evaluaton Toolkit:

1. Download this github file:

```bash
git clone https://github.com/wjc0602/TNL2K_evaluation_toolkit_python
```

2. Unzip related files for evaluation:

```bash
cd annos && unzip ./annos.zip
```

3. Place the tracking results you need to evaluate in the tracking_results folder and modify the file.
   file.

```bash
utils/config_tracker.py
```

4. Open the Matlab and run the script:

```bash
python evaluate_tnl2k_dataset.py
```

5. Wait and see final results in result_fig folder.

## Acknowledgement

This code is modified based on the MatLab version evaluation toolkit of [[TNL2K](https://github.com/wangxiao5791509/TNL2K_evaluation_toolkit)].

If there is any infringement involved, please contact me to delete it via wjcmain@gmain.com.