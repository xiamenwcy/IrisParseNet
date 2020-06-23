Usage:
After obtaining the predicted iris mask(data/mask), pupil mask(data/pupil) and iris outer boundary(data/iris) from the proposed network, the post-processing is very easy.
First, the config/Config.ini is required to set. Then enter the path of "Release/IrisParseNet.exe" from the cmd.exe in Windows.  
Finally, run the command such as:

IrisParseNet.exe -h
Usage: IrisParseNet.exe [options]
Usage: This project is mainly used for joint iris segmentation and localization.
 Just Enjoy it!

Options:
  -?, -h, --help                      Displays this help.
  -v, --version                       Displays version information.
  -c, --config <default config path>  Set config file path [default:
                                      ../../config/Config.ini].


In the final results, we have offered the fitted inner and outer boundaries of the iris (data/new_iris_circle, data/new_pupil_circle),
refined iris mask(data/new_mask_circle), the localization parameters(data/seg_param_circle), the localization and normalization results(data/seg_circle).
Our post-processing supports the circle and ellipse fitting, see Config.ini. 