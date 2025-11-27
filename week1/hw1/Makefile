all:	hw1prob1.pdf

hw1prob1:	hw1prob1.c
	gcc -o hw1prob1 hw1prob1.c -lm

hw1prob1.txt:	hw1prob1
	./hw1prob1 > hw1prob1.txt

hw1prob1.pdf: hw1prob1.txt hw1prob1.py
	python3 hw1prob1.py
	rm -f hw1prob1 hw1prob1.txt
	
# Clean intermediate files, keep PDF
clean:
	rm -f hw1prob1 hw1prob1.txt
