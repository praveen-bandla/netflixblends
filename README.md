# Netflix Blendz


## **Introduction**

***Welcome to Netflix Blendz!***

Have you and your friends ever struggled to find something to watch together on Netflix? If the answer is yes, then keep reading to learn more about our project **Netflix Blendz**! We think you'll find it useful...

*Who are we*: We are a group of students who created Netflix Blendz as a way to help indecisive friends pick movies from Netflix to watch together. 

*What is a Netflix Blend*: Netflix Blendz is a webapp that allows two users to upload their Netflix user watch history to generate a custom list of recommended movies titles that they can enjoy together. A blend of your viewing tastes if you will. 

*How does it work*: Keep reading to find out.


In this blog post, we will share the process of how we created this product and some of the technical aspects that were involved! Some components we will talk about are `complex data visualization`, using `SQL databses`, and building a `web application`. 

<br>

## **Project Overview**

To implement this project, we compiled a master list of all Netflix titles to choose from, performed user-based collaborative filtering to generate recommendations, and ultimately create a webapp with numerous data visualizations. As such, there were 4 broad technical components in this project:


1.   **Data Collection and Processing**: Maintain a Netflix Library dataset and process user data for subsequent analytics
2.   **Visualizations**: Provide users with insights into their watch behavior with meaningful visualizations
3. **Webapp**:
    1. **Building a Dynamic Website**: Create a a dynamic website that users can navigate through to find their blend
    2. **Database Management**: Develop an efficient data storage architecture that enables quick storage and retreival of submitted user data
4. **Recommender Sytem**: Generate list of movies for users to watch using sophisticated machine-learning methods

![img0](https://user-images.githubusercontent.com/114946455/235496959-ff467f56-a595-4ddc-bf28-46f3bd9eac84.png)

<br>

## **Data Processing**

Our first main goal, naturally, was to ensure all our data is usable and clean. In particular, the two sources of data that we consider are:

1. **Netflix Library Dataset**: A comprehensive dataset of all titles on Netflix US used to generate insights
2. **User Watch History Data**: A dataset of all titles watched by a user, submitted through the webapp

<img width="445" alt="img1" src="https://user-images.githubusercontent.com/114946455/235496552-5aebbb36-9792-4cf6-9eb6-2f0ba071a3df.png">


As evident in a sample user watch history data snippet above, extracting the titles from the user history requires some cleaning. In particular, we performed string splitting on the colon character `:` to extract the titles and categorize them as movies or TV shows. However, as one can imagine, this did not always produce the desired result. Two notable such instances are when TV episode titles contain the colon character `:` and when TV show titles aren't stored in the assumed format:

<img width="767" alt="img2" src="https://user-images.githubusercontent.com/114946455/235496707-a78d985f-1c93-405c-a6e8-338e94ade6c3.png">
<img width="765" alt="img3" src="https://user-images.githubusercontent.com/114946455/235496722-10f94d43-2d1e-4e26-ae52-684d553f0d81.png">


However, when merging this data with our Netflix library, we did not observe a signficant loss of data from these cases. Thus, we have treated them as edge cases that would require manual effort.

On the other hand, our Netflix library data was sourced from a [Kaggle dataset](https://www.kaggle.com/datasets/victorsoeiro/netflix-tv-shows-and-movies), which contains credits and cast information from all titles on Netflix as of July 2022. Although this certainly was not our first choice, we ultimately stuck with this dataset simply due to a lack of other alternatives. Our initial attempt of sourcing data from [whats-on-netflix.com](https://www.whats-on-netflix.com/library/?utm_medium=rightside&utm_source=whats-on-netflix) yielded data that required a substantially greater amount of cleaning and contained too many missing values for our liking.

<br>

## **Visualizations**
*Technical Component: Complex data visualization*

Visualizations that were ultimately embedded in the webpage were created with Plotly, as we desired interactive plots that users could explore. Next, we converted our plots to JSON using `json.dumps()` and the JSON encoder that comes with Plotly. We did this since the webpage uses the Plotly Javascript library to render the plot, which requires the Plotly chart encoded as JSON. To enable the `insights.html` page to dipslay these plots, we then wrote a short script which sets a variable to the imported JSON code and then calls the plot method from the Plotly library to display the chart. The result is a set of visualizations similar to the ones generated below:


<img width="1440" alt="img4" src="https://user-images.githubusercontent.com/114946455/235496770-764bfa08-563e-47b8-bc77-7caf3a74c91f.png">

<br>

## **Webapp**
*Technical Components: Building a dynamic website, Creating and interacting with a SQL database*

### **Building the Website**

The webapp, screenshots of which are displayed below, was built using Flask. The bulk of the back-end work was incorporated into our `app.py` file, which contains the core functions used to build and run the dynamic webpages. In particular, the file contains all functions required to render each page's `.html` file, which were written separately.

The hard-coded contents of each page, including stylistic features, were incorporated in each page's `.html` file and the `style.css` file. The `base.html` file was used to build a skeleton template that all pages inherited. Observe that it includes hyperlinks to the home page embedded in the titles/logo, as well as a navigation bar to easily switch between the different pages.

In turn, this creates webpages that look like:

![netflix1](https://user-images.githubusercontent.com/114946455/235497396-49509695-a92c-4eae-bf2f-2a994dd45e8e.jpg)
![netflix2](https://user-images.githubusercontent.com/114946455/235497406-55b26eed-db38-4318-bdef-2a3d686d710f.jpg)
![netflix3](https://user-images.githubusercontent.com/114946455/235497411-e83d8315-aed1-46d2-b52d-835196924e5e.jpg)


As you may be thinking to yourself, these webpages clearly aren't too...sophisticated. When building the Flask infrastructure, we spent much of our time developing the functionality of it, leaving us with little time and resources to develop the design. If there is a further expansion of the project, we hope to improve the aesthetic of the webpage! As things stand, it served its purpose and enabled us to acheive our goals!

<br>

### **Database Management**

One of the challenges we faced when creating this project was figuring out how to manage all of the user data. After a user uploads their csv file to the webpage, we had figure out a way to assign a unique id to each user’s csv and store them in some type of database that could be accessed by anyone using our web app. To access, manage and store our viewing history data, we decided to use a SQL database. Our database holds tables, each with an unique id name, that would map to a specific user’s csv file. 

To implement this, we wrote three functions: `get_user` to create the database and tables, `insert_history` to populate the tables with information from the csv files, and `get_history` to retreive the correct user file and prepare it for the recommendation system. These functions use the SQLite package from Python to connect to and query our database. The user-uploaded csv files are first converted into a pandas dataframe, then transformed into SQL tables. Here is a visual diagram with written explanations for how these functions work:

![imgdatabase](https://user-images.githubusercontent.com/114946455/235497047-fa4964b1-0864-4a7f-ba8d-61afb8d6e856.jpg)

<br>

## **Recommendation System**

### Brainstorming

Building a recommendation system is never easy. Over the course of working on this project, we had been through several working models of what the recommendation engine could look like. Many of our ideas did not pan out. There were several factors we had to take into account. These are:

1. **Time**:  Because this is a webapp, we needed to ensure that the generation of the blend would not take too long! This meant, that our recommendation engine needed to run and process quickly. By far, this was the most challenging variable to account for.

2. **Simplicity**:    We needed to ensure that our system was thorough enough to account for many different variables changing and thereby including many edge cases. For example, what if one user had significantly more data than the other? What if two users had vastly differing viewing histories? In other words, we needed to make sure we did not make too many assumptions of the data fed to the model

3.  **Complexity**:   Movies are complex structures to understand which, needless to say, necessitate many dimensions of modelling. We needed to create an algorithm that would account for said complexity and thoroughly understand the different features movies present in order to create strong recommendations

<br>

### Movie Lens Dataset
In the vast array of resources that exist in on the internet, we were confident we could find a solution! And we did, through the `movielens_data_genome` dataset. Here is a summary of the text in the `readme` of the downloaded dataset:

*This dataset contains raw movie data for generating the Tag Genome dataset and results of the experiments conducted in [Kotkov et al., 2021]. The Tag Genome dataset contains movie-tag pair scores which indicate the degrees to which tags apply to movies. This dataset was introduced in [Vig et al., 2012]. To generate the dataset, the authors collected the following information about movies: metadata (title, directors, actors...), movie reviews from IMDB (https://imdb.com/), ratings, tags that users attached to movies in MovieLens (https://movielens.org/) and user judgements regarding the degrees to which tags apply to movies. The authors collected user judgements with a survey in MovieLens and used these data to train and evaluate their algorithm. The algorithm is based on regression and predicts the movie-tag scores. The authors of [Kotkov et al., 2021] prepared the raw movie data for publication, refactored the programming code of the algorithm and introduced TagDL, a novel algorithm for prediction of movie-tag scores. TagDL is based on neural networks and uses the same features as the regression algorithm of [Vig et al., 2012]. The code is available in the GitHub repository via the following link: https://github.com/Bionic1251/Revisiting-the-Tag-Relevance-Prediction-Problem*

Essentially, the dataset uses the model architecutre designed in two different experiments along with certain tag metrics detailed by the very same to make understandings of different movies. There are about 1000 different tags which categorize each movie based on different parameters. These include, 'historic', 'feel-good', 'anime', 'sports', etc. In each of the experiments, the authors created designed a model architecture for optimally predicting truth values for each category for each movie based on modelled survey data. What MovieLens did was take that a step further, collecting IMDB reviews for each of around 50,000 movie titles to predict/generate label values for each title. They uploaded label results for about 10,000 of the titles, and collected reviews for many more!

We had the model architecture used to generate more of the labels, avaialble through the Github page of the developers. However, the review dataset and the model implementation proved to be extremely complicated, and we were thus not able to train the model on the dataset! That being said, we felt 10,000 titles was plenty, so we decided to use that as the basis of our algorithm.

<br>

### The algorithm

Our job is not close to done! We have label values for 10,000 movies on 1000 parameters, but we do not yet have a recommendation system. On a high level, what we did is the following:

1. Collect User Data
2. For each movie watched, concat the label values
3. Aggregate the data to generate a 'preference vector' - essentially a value of preference for each label
4. Using the preference vector, compare the user's interest to the label values of each movie, generating an interest metric
5. Aggregate the interest metric for each movie
6. Repeat 1-5 for user2
7. Create a joint ordering of all movies based on the interest metrics of each title for user 1 and user 2


#### 1. Collecting user data

This has been explored in previous sections

#### 2. Concating the label values for a user's watch history

This was implemented using the `generate_user_history_tag`
Essentially, if a user watched,

$$ u1 = [m_1, m_2, \dots, m_n] $$

And each movie had the labels:

$$ mi = [l_{i,1}, l_{i,2},\dots, l_{i,1000}] $$

Then, the function `generate_user_history_tag` would create the following matrix:

```math
\begin{equation*}
mat1 = 
\begin{pmatrix}
l_{1,1} & l_{1,2} & \cdots & l_{1,1000} \\
l_{2,1} & l_{2,2} & \cdots & l_{2,1000} \\
\vdots  & \vdots  & \ddots & \vdots  \\
l_{n,1} & l_{n,2} & \cdots & l_{n,1000} 
\end{pmatrix}
\end{equation*}
```

#### 3. Aggregate the matrix to generate a 'preference vector'

Here, we use a function to aggregate each label value across the movies.
I.e,

$$ L_{i} = f([l_{1,i}, l_{2,i}, \dots , l_{n,i}]) $$

Repeating for each value, we have:

$$ L = [L_{1}, L_{2}, \dots, L_{1000}] $$

This was implemented using the function `generate_user_pref_vector()` with `f` as the function.
This was simply defined to be:

```math
For [l_{1,i}, l_{2,i}, \dots, l_{n,i}],
L_{i} = ( \sqrt{l_{1,i}} + \sqrt{l_{2,i}} + \dots + \sqrt{l_{1,i}} )^2
```


#### 4. and 5. Generating an interest metric and aggregating

This is done by finding the euclidean distance between $L =[L_{1}, L_{2}, \dots, L_{1000}]$ for the user and the label vector for each movie. Applied using the function `generate_user_movie_interest`.

In code, this was easier done using $L =[L_{1}, L_{2}, \dots, L_{1000}]$ as a vector and a master matrix with the labels for each movie, just like mat1, except for all movies. The result is a vector:

$$ I = [I_{1}, I_{2}, \dots, I_{10,000}] $$

As there were 10,000 movies. This was the final vector for the user generated. Now, we repeat for user 2!


#### 6. Repeat for user2.

Repeated for user 2



#### 7. Ordering the results

Note that $L$ contains a value from 0 to 1 quantifying the interest value for each user of each title! To generate the recommendation for 1 user, all we do is sort by interest value. For two users, it gets more tricky. We want to find titles that both users are interested in. What we did is generate the average of each users interest for a title, and sort! In the function `generate_recommendation` that does this, we accept `n` as a parameter, indicating the number of movies that the user would like to be recommended. What we first did is filter for the first `2n` values based on the highest average. Within those `2n` entries, we collected the first `n` with the lowest standard deviation. This way, we balance both users interest, opposed to simply allowing one user's to dominate.


And with that, we are done! That was a lot, we hope it made some sense! Here is how the results could look!

![blog_output](https://user-images.githubusercontent.com/114946455/235497506-267f33a7-1f0a-4b48-ab7f-373833ee1304.jpg)

We created a function called `generate_two_user_recommendation` that compiles all these functions neatly and generates a list of recommended output. Note that we did not have all 50,000 titles in the movie lens dataset (which isn't even to say all titles are on Netflix, and vice versa), but only 10,000, so we generated the recommendations limited to those movies alone! The final function returns a list of all movies watched by the users that were not in the dataset. What we noticed is that this included many international titles.

<br>

## **Concluding Remarks**

**Ethical Ramifications**: *When creating any project for the public, it is important to consider the ethical ramifications that may result from it. In this section, we will discuss some of these ramifications.*

- Creating a recommendation system can allow for individuals to make inferences about user behavior which, in some cases, can lead to privacy concerns. For example, some people may have titles in their watch history that they consider to be embarrassing, or would simply prefer to keep private. In our current method of uploading csv files with user history, there is no system to filter out unwatned titles from being seen by the recommender. This may result in unwanted or private information being shared with the other person involved in the blend. If this is a concern that you have, do not worry. Netflix has an option in settings that allows you to hide specific titles from viewing history. If there is something on there that you would prefer to keep private when making a blend, simply hide that title from your history!

- Another concern to consider is the potential for a recommendation system to generate titles with bias towards a particular political view or perspective. Since Netflix has many titles regarding social justice issues and politics, recommending shows or movies that mock, dehumanize or marginalize minority groups would be extremely harmful and inappropriate. We recognize that this is a complex issue, however, under the assumption that generated recommendations should appeal to everyone in the user group, it is not likely offensive titles will make it to the list of recs. 

**Last Words**

Despite the ethical ramifications we have listed above, overall we do believe that Netflix Blendz will spark more joy than concern for our users (assuming of course that people would benefit from spending more time with their friends on Netflix). With that, we hope you found this blog post useful/interesting and are willing to give our product a try. 

Happy viewing!
-- Joy, Sid and Praveen
