# Neuroimaging-ITM

Implementation of the papers
Neuroimaging-ITM: A Text Mining Pipeline Combining Deep Adversarial Learning with Interaction Based Topic Modeling for Enabling the FAIR Neuroimaging Study.


## Task
Given a sentence:

1. give the entity tag of each word (NER) 

2. the relations between the entities in the sentence. 

The following example indicates the accepted input format of our model:


```
#doc 0144858-13
0	0144858	O	['N']	[0]
1	:	O	['N']	[1]
2	introduction	O	['N']	[2]
3	neural	O	['N']	[3]
4	processing	O	['N']	[4]
5	in	O	['N']	[5]
6	the	O	['N']	[6]
7	ventral	O	['N']	[7]
8	visual	B-SEN	['SEN-COG']	[14]
9	pathway	O	['N']	[9]
10	is	O	['N']	[10]
11	fundamental	O	['N']	[11]
12	for	O	['N']	[12]
13	object	B-COG	['N']	[13]
14	recognition	I-COG	['N']	[14]
15	in	O	['N']	[15]
16	primates	O	['N']	[16]
17	.	O	['N']	[17]
```

## Run the model

```
./run.sh
```

