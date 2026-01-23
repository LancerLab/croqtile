To run the compiled Choreo program on real hardware like GCU, the current Makefile can assist with the environment configuration. The command:
```
make setup-gcu2
```
sets up the GCU-2.x compiler and runtime environment. Additionally,
```
make gcu2-kmd
```
helps configure the GCU-2.x hardware driver.

Similarly, the commands:
```
make setup-gcu3
```
and
```
make gcu3-kmd
```
assist in setting up the GCU-3.x compiler, runtime, and hardware driver.

