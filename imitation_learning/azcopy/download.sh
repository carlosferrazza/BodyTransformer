file=small.tar.gz # use the empty string "" to download everything in the blob

# Source URL can be copied from "Download Dataset" in Microsoft Research page:
# https://www.microsoft.com/en-us/research/publication/mocapact-a-multi-task-dataset-for-simulated-humanoid-control/
# Note the presence of `${file}$` in the middle of `src` and backslashes `\` before `?`, `=`, and `&`.
src=https://msropendataset01.blob.core.windows.net/motioncapturewithactionsmocapact-u20220731/${file}\?sp\=rl\&st\=2023-08-22T17:20:43Z\&se\=2026-07-02T01:20:43Z\&spr\=https\&sv\=2022-11-02\&sr\=c\&sig\=L9f3Y8jAz3SCoAM5U5g9uCs0pmTI40rDLh2ZGC7OxE8%3D

./azcopy copy $src $1 --recursive
