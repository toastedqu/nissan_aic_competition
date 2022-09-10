array=(back6
street2
)

mkdir images
for var in ${array[@]}
do
    mkdir "images/${var}" 
    ffmpeg -i "${var}.mp4" -vcodec png "images/${var}/%08d.png"
    echo    ffmpeg -i "${var}.mp4" -vcodec png "${var}/%08d.png"
done
