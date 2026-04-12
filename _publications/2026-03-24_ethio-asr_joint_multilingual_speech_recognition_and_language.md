---
title: "Ethio-ASR: Joint Multilingual Speech Recognition and Language Identification for Ethiopian Languages"
date: 2026-03-24
paper: http://arxiv.org/abs/2603.23654v1
authors: "Badr M. Abdullah, Israel Abebe Azime, Atnafu Lambebo Tonja, Jesujoba O. Alabi, Abel Mulat Alemu, Eyob G. Hagos, Bontu Fufa Balcha, Mulubrhan A. Nerea, Debela Desalegn Yadeta, Dagnachew Mekonnen Marilign, Amanuel Temesgen Fentahun, Tadesse Kebede, Israel D. Gebru, Michael Melese Woldeyohannis, Walelign Tewabe Sewunetie, Bernd Möbius, Dietrich Klakow"
layout: default
---

We present Ethio-ASR, a suite of multilingual CTC-based automatic speech recognition (ASR) models jointly trained on five Ethiopian languages: Amharic, Tigrinya, Oromo, Sidaama, and Wolaytta. These languages belong to the Semitic, Cushitic, and Omotic branches of the Afroasiatic family, and remain severely underrepresented in speech technology despite being spoken by the vast majority of Ethiopia's population. We train our models on the recently released WAXAL corpus using several pre-trained speech encoders and evaluate against strong multilingual baselines, including OmniASR. Our best model achieves an average WER of 30.48% on the WAXAL test set, outperforming the best OmniASR model with substantially fewer parameters. We further provide a comprehensive analysis of gender bias, the contribution of vowel length and consonant gemination to ASR errors, and the training dynamics of multilingual CTC models. Our models and codebase are publicly available to the research community.