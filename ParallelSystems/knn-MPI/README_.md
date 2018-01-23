# knn-MPI

### How to run the project:

1. Change to root directory of the project:

    ```
    $ cd /knn-MPI
    ```

2. Compile source code:

    ```
    $ make
    ```
3. Run the desired implementation (1.Serial, 2. MPI blocking, 3. MPI non-blocking)

```
    $ ./bin/main 1 600000 30 30               //serial

    $ mpirun -np 4 ./bin/main 2 60000 30 30   //MPI blocking

    $ mpirun -np 4 ./bin/main 3 60000 30 30   //MPI non-blocking
```
### Change corpus and validation .txt file:

1.  Place the new .txt files at the **_./data_** directory
    
2.  Edit their paths at **_./include/global_vars.h_** header file
    ```
    $ nano /include/global_vals.h
    ```
3.  Clean project and re-compile 
    ```
    $ cd /knn-MPI
    $ make clean
    $ make
    ```
4. Run the project :+1:
