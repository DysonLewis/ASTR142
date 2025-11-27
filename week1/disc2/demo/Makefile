all: hello pydemo

helloworld: helloworld.c
	gcc helloworld.c -o helloworld

hello: helloworld
	./helloworld

pydemo: demo1.py demo2.py
	python demo1.py &
	python demo2.py

clean:
	rm helloworld
	rm -f *.pdf
	rm -f *.dat
