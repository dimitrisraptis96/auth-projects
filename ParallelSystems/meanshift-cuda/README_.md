# meanshift-cuda

### How to run the project:

1. Change to root directory of the project:

    ```
    $ cd /meanshift
    ```

2. Compile source code:

    ```
    $ make
    ```
3. Run the desired implementation (1.Serial, 2. Cuda shared, 3. Cuda non-shared)

```
    $ ./bin/meanshift 1 dataset/txt/5000_2.txt 100000 5000 2    //serial

    $ ./bin/meanshift 2 dataset/txt/5000_2.txt 100000 5000 2    //cuda shared

    $ ./bin/meanshift 3 dataset/txt/5000_2.txt 100000 5000 2    //cuda non-shared
```