echo "Help me debug the following:" > prompt.txt
echo "" >> prompt.txt
cat custom_op.cc >> prompt.txt
echo "" >> prompt.txt
echo "//src/lib.rs" >> prompt.txt
cat src/lib.rs >> prompt.txt
echo "" >> prompt.txt
echo "//main.py" >> prompt.txt
cat main.py >> prompt.txt