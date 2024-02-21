<h1 align="center">Netflix-Recommendation-System</h1>
<p><font size="3">
A web-app which can be used to get recommendations for a series/movie, the app recommends a list of media according to list of entered choices of movies/series in your preferred language using <strong>Python</strong> and <strong>Streamlit</strong>.
</p>

 # This web-app contains 3 main pages:
- [Home Page](#home-page) with Recommendations
- [Movie Detail Page](#movie-detail-page)  with Recommendations
- [Netflix Page](#netflix-page)

## Home Page with Recommendations
Here the user can choose list of their favourite movies and series and their preferred language. For example, I have entered a list with 2 Horror Movies(Insidious and Insidious Chapter 2), an action series(Supergirl) and a drama series(Suits) as my list of choices and English and Hindi as my preferred languages.
Clicking on the Get Recommendation button the user will get poster images of all the recommended movies and series sorted based upon their IMDb Scores.

Clicking on any poster image, the user will be sent to the Movie Details page for the corresponding title.

![](/Netflix Recommendation System/UI_SS/HomePage.png)
![](/Netflix Recommendation System/UI_SS/HomePage With Selected.png)
![](/Netflix Recommendation System/UI_SS/HomePage with Recommendation.png)
![](/Netflix Recommendation System/UI_SS/HomePage with Recommendation1.png)

## Movie Detail Page with Recommendations
Here are the complete details of the user selected title like Genre, Movie Summary, Languages in which movie is available, IMDb scores, Directors, Writers and Actors and so on. User will also find a link at the end of the page for the NEtflix Page of the corresponding title. 
Here also the user will get poster images of all the recommended movies and series based on the selected movie i.e. the movie whose details you are watching.

Clicking on any poster image, the user will be sent to the Movie Details page for the corresponding title.

![](/Netflix Recommendation System/UI_SS/Moviepage Selected Movie.png)
![](/Netflix Recommendation System/UI_SS/Moviepage Recommendation.png)

## Netflix Page
This page is not a part of my web-app but an example what the user will see as the Netflix Page if they choose to click on the Netflix Link for the title.
You can login into your Netflix account and enjoy watching your selected movie or series from our recommendations.
![](/Netflix Recommendation System/UI_SS/Netflix Page.png)

# How To Use

To be able to use this web app locally in a development environment you will need the following:

1) You will need [Git](https://git-scm.com) installed on your computer.

2) Then From your terminal, you should do the following:

```cmd
# Clone this repository
git clone https://github.com/Andy-16-18/Netflix-Data-Analysis-and-Recommendations-System.git

# Go into the repository
cd netflix-recommendation-system

# Install Streamlit (if you already haven't)
pip install streamlit

```
3) To run this application you don't need to have any special configuration but make sure you don't change the directory of the project otherwise you can recieve errors while you try to run the app.

4) You can run the Netflix Recommendation System using the following command from your terminal:

```
# Open the Command Promt
# Install Streamlit (if you already haven't)
>> pip install streamlit
# Then Navigate to the folder where main.py file is present
>> cd filepath
# Then Enter the following Command in the command promt
>> streamlit run main.py
```

# Author

👤 **Andy**
- Github: https://github.com/Andy-16-18/

# Show Your Support 

Give a ⭐️ if you like this project!
