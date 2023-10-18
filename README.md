# CA_Assignment1_P2

This repository contains python script to generate Linear regression model for creating CPI stack for 4 spec benchmarks.

This is part of Assignment 1 P2 for coures E0 243 Computer architecture offered at CSA department at Indian Institute of Science

## How to use it
We added csv files of data points for all 4 benchmarks which we generated using perf. 
To run the linear regression model use following command:
``` 
python3 Assignment1_p2.py <name of the csv file>
```

It will print the cofficients and qualitative matrics for model. Residual graph will be saved in png file.