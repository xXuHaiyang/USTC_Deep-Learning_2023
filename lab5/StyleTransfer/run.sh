# for all content pictures in contents/, run on all style pictures in styles/
for content in contents/*.jpg; do
    for style in styles/*.jpg; do
        python main.py -c $content -s $style -e 300 -i 256
    done
done