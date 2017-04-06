This project is an analysis on Billboard Top 100 Track in year 2000. I got a raw data of Billboard Top 100 tracks, artists, genre, dates and their ranks from year 2000. I wanted to take a closer look at how track categories (genre) performed from each other. I came up hypothesis to see if genre make a significant difference in terms of the length of days on Billboard as well as the length between peaked dates and entered dates. This result might have an overall view on the popular music trend in 2000.

Here are the main steps I took:
- Get data: a quick scan of the data in order to identify assumptions and problems
- Clean data: based on the uncovered problems, take action on cleaning the dataset for futher analysis
- Analyze data: plot on major variables and find insights related to the hypothesis
- Test hypothesis: use t statistics to test the statements and summarize the results

## Get Data
1. Use pd.read_csv to read data
- The data has 317 rows and 83 columns

2. Identify the risk and assumptions:
- There might be mistakes on entering the data as the table contains an amount of numbers
- I assume the date peaked is the date when the track had the highest rank on Billboard
- Some tracks had entered date before 2000, but we assume all data was captured for 2000 only

3. A quick look at the raw data by using functions such as .info, .value_counts, .unique etc. The followings are some problems I found:
- There are a lot of '*' in week columns due to non-exsiting data
- The format in Time column is not appropriate
- The format for rank numbers are all objects
- Column 'genre' has multiple formats for the same genre, such as 'R&B' and 'R & B'.


## Clean Data
Before I go further to my analysis, I spent quite an amount of time on cleaning data. In the meantime, I also found new problems and added them to the list. Over and over, the dataset has become easier to work on.

1. Create a function to convert * to NaN and used applymap to the dataframe
- By doing this, I noticed some columns at the end only contains NaN data. I decided to remove - use .unique function to exam columns at the end. It turned out that all columns after 'x65th.week' can be removed.

2. Clean column 'time', 'date.entered' and 'date.peaked'
- Use str.replace to change ',' to ':'
- Use .stripe to cut off 'AM' as all times were in 'AM'
- Use to_datetime.dt.time to change the time format without adding the date
- Use tp_datetime to convert all data for 'date.entered' and 'date.peaked' columns

3. Clean week columns
- Use .applymap and lambda to convert all strings here to float

4. Clean column 'genre'
- Similarly use .applymap and lambda to make entries more consistent i.e. use 'R&B' for R&B' and 'R & B', use 'Rock' for 'Rock' and 'Rock'n'roll'


## Analyze Data
Since I am more interested in the time a track stayed on Billboard, as well as the time between it entered and peaked. I would create 2 new columns that contain integers for my analysis.

1. Create a column 'diff.between.peaked.and.enetered'
- Substracting 'date.entered' from 'date.peaked'
- In order to use the column on plotting, I use slice function to cut off ' days' and converted them into integers

2. Create a column 'no.of.weeks.on.billboard'
- The logic here is to count the columns with valid numbers
- Slice the dataset to week columns only
- Use a for loop to count each row and append them into a new list
- Add the list into the dataframe as a new column

3. Plot histograms on new columns

4. Understand relationship between new columns
- Plot a scatter and best fit linear regression line

5. Pivot table on index genre column for next step


## Test Statements and Next Steps
In this step, I came up with the null hypothesis and alternative hypothesis for each statement. I only looked at the music categories that had the most entries: Rock, Country and R&B.

In my next steps, I would like to
- Examine genre column to see if there is anythig misentered
- Use a for loop to calculate pvalue for each pair combination


