# 🏛️ Vesuvius Challenge - Ink Detection
<img src='assets/header.png' alt=''>

## 📜 Project Description

This exciting competition, available on Kaggle and supported by the Vesuvius Challenge organization, aims to bring back to life an ancient library buried under the ashes of a volcano. In this competition, our mission was to detect ink from 3D X-ray scanners. Thousands of scrolls were part of a library located in a Roman villa in Herculaneum. This villa was buried by the eruption of Mount Vesuvius nearly 2000 years ago. The scrolls were charred by the heat of the volcano and are now impossible to open without damaging them.

## 📷 Tilling Method

<img src='assets/tilling.gif' style='width: 60%' alt='Ink labels with tilling method performing on it'>

In order to process the ultra-high definition images effectively, we employed an image tiling method. The images were divided into smaller sub-images with a size of 256 by 256 pixels to feed into the model. This approach allowed us to handle the large dataset more efficiently. Additionally, we implemented a selection criterion where we only considered tiles that contained a minimum of 5% ink pixels. By focusing on these specific tiles, we were able to concentrate our efforts on the areas most likely to contain valuable information within the ancient scrolls.

## 🤖 Model architecture

<img src='assets/model_architecture.png' style='width: 80%' alt='Architecture of EficientUnet V2 model'>

## Contributors

Louis REBERGA <a href="https://twitter.com/rbrgAlou"><img src="https://abs.twimg.com/favicons/twitter.3.ico" width="18px"/></a> <a href="https://www.linkedin.com/in/louisreberga/"><img src="https://static.licdn.com/sc/h/akt4ae504epesldzj74dzred8" width="18px"/></a> <a href="louis.reberga@gmail.com"><img src="https://www.google.com/a/cpanel/aqsone.com/images/favicon.ico" width="18px"/></a>

Baptiste URGELL <a href="https://twitter.com/Baptiste2108"><img src="https://abs.twimg.com/favicons/twitter.3.ico" width="18px"/></a> <a href="https://www.linkedin.com/in/baptiste-urgell/"><img src="https://static.licdn.com/sc/h/akt4ae504epesldzj74dzred8" width="18px"/></a> <a href="baptiste.u@gmail.com"><img src="https://www.google.com/a/cpanel/aqsone.com/images/favicon.ico" width="18px"/></a> 

Johan MONCOUTIÉ <a href="https://twitter.com/moncoutiej"><img src="https://abs.twimg.com/favicons/twitter.3.ico" width="18px"/></a> <a href="https://www.linkedin.com/in/johan-moncoutie/"><img src="https://static.licdn.com/sc/h/akt4ae504epesldzj74dzred8" width="18px"/></a> <a href="johan6446@gmail.com"><img src="https://www.google.com/a/cpanel/aqsone.com/images/favicon.ico" width="18px"/></a>

