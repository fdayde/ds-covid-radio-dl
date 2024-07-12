COVID Lung X-Rays Classification with Deep Learning
==============================
This project is a fork of [MAR24_BDS_Radios_Pulmonaire](https://github.com/DataScientest-Studio/MAR24_BDS_Radios_Pulmonaire) which was developped during the Data Scientist course of [Datascientest](https://datascientest.com/) from March to June 2024.  

The primary goal of this fork is to introduce new features and improvements to enhance functionality, the user interface and performance. 

------------
### Improvements
- [x] Added references for dataset, models and images.
- [x] Provided context of the project on the homepage.
- [ ] Redesigned the footer.
- [x] Light code refactoring.
- [ ] Added some lung x-rays images for testing.
- [ ] Introduced a lung segmentation model.
- [ ] Deep code refactoring?
------------
View the updated **streamlit app** on [Hugging Face](https://huggingface.co/spaces/fdayde/streamlit-dl-radio) 🤗

------------
### How to deploy the streamlit app on Hugging Face: 

- Create a new space on Hugging Face and clone the repository
- Push the content of the `src/streamlit` directory
- Add the model's weights file to the `models` folder
- Store the model weights in Git LFS by adding the following line to the `.gitattributes` file:  
```*.h5 filter=lfs diff=lfs merge=lfs -text```
- Push to Huggingface
- Do not modify or delete the `REAMDE.md` file created by Hugging Face during the initialization on the space.

------------
### Dataset

The COVID-QU-Ex dataset is available on Kaggle: https://www.kaggle.com/datasets/anasmohammedtahir/covidqu


[1] A. M. Tahir, M. E. H. Chowdhury, A. Khandakar, Y. Qiblawey, U. Khurshid, S. Kiranyaz, N. Ibtehaz, M. S. Rahman, S. Al-Madeed, S. Mahmud, M. Ezeddin, K. Hameed, and T. Hamid, “COVID-19 Infection Localization and Severity Grading from Chest X-ray Images”, Computers in Biology and Medicine, vol. 139, p. 105002, 2021, https://doi.org/10.1016/j.compbiomed.2021.105002.

[2] Anas M. Tahir, Muhammad E. H. Chowdhury, Yazan Qiblawey, Amith Khandakar, Tawsifur Rahman, Serkan Kiranyaz, Uzair Khurshid, Nabil Ibtehaz, Sakib Mahmud, and Maymouna Ezeddin, “COVID-QU-Ex .” Kaggle, 2021, https://doi.org/10.34740/kaggle/dsv/3122958.  

[3] T. Rahman, A. Khandakar, Y. Qiblawey A. Tahir S. Kiranyaz, S. Abul Kashem, M. Islam, S. Al Maadeed, S. Zughaier, M. Khan, M. Chowdhury, "Exploring the Effect of Image Enhancement Techniques on COVID-19 Detection using Chest X-rays Images," Computers in Biology and Medicine, p. 104319, 2021, https://doi.org/10.1016/j.compbiomed.2021.104319.  

[4] A. Degerli, M. Ahishali, M. Yamac, S. Kiranyaz, M. E. H. Chowdhury, K. Hameed, T. Hamid, R. Mazhar, and M. Gabbouj, "Covid-19 infection map generation and detection from chest X-ray images," Health Inf Sci Syst 9, 15 (2021), https://doi.org/10.1007/s13755-021-00146-8.  

[5] M. E. H. Chowdhury, T. Rahman, A. Khandakar, R. Mazhar, M. A. Kadir, Z. B. Mahbub, K. R. Islam, M. S. Khan, A. Iqbal, N. A. Emadi, M. B. I. Reaz, M. T. Islam, "Can AI Help in Screening Viral and COVID-19 Pneumonia?," IEEE Access, vol. 8, pp. 132665-132676, 2020, https://doi.org/10.1109/ACCESS.2020.3010287.

------------
### Original Project: 

- Thomas Baret [linkedin](https://linkedin.com/in/thomas-baret-080050107) [github](https://github.com/tom-b974)
- Nicolas Bouzinbi [linkedin](https://linkedin.com/in/nicolas-bouzinbi-7916481b4) [github](https://github.com/NicolasBouzinbi)
- Florent Daydé [linkedin](https://linkedin.com/in/florent-daydé-16431469) [github](https://github.com/fdayde)
- Nicolas Fenassile [linkedin](https://linkedin.com/in/nicolasfenassile) [github](https://github.com/NicoFena)

supervised by: Gaël Penessot

View the original streamlit app on [Hugging Face](https://huggingface.co/spaces/fdayde/streamlit-dl-radio) 🤗


<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
