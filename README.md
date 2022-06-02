# Algoritmus na zapisovanie a vyhodnocovanie produktovej superpozície
(Bakalárska práca)

Program je implementovaný v jazyku Python a pre podporu hlbokého učenia bola zvolaná
distribúcia Anaconda. Tento balíček už obsahuje Python 3, rôzne knižnice pre strojové uče-
nie a ďalšie pomocné funkcie. Prostredie, v ktorom budeme v tomto manuáli pracovať je
PyCharm Professional, hlavne kvôli jednoduchému spôsobu prepojenia s verzovacím sys-
témom GitHub. Databázový systém využívaný v implementačnej časti je Microosft SQL
Server 2019 spolu s jazykom T-SQL.


Postup inštalácie:
1. Na stránke https://www.anaconda.com/products/distribution stiahnuť distri-
búciu Anacondy vo verzii 3.9.
2. Na stránke https://www.jetbrains.com/pycharm/download/ stiahnuť vývojové
prostredie PyCharm vo verzii Professional. Ak z akéhokoľvek dôvodu verzia nejde
stiahnuť, postačí aj Community verzia.
3. Na stránke https://www.microsoft.com/en-us/sql-server/sql-server-downloads
stiahnuť SQL server v developer verzii.
4. Spustiť inštaláciu SQL servera a na konci inštalácie naištalovať aj prostredie SSMS
podľa pokynov. Na nastaveniach netreba nič meniť.
5. Spustiť inštaláciu Anacondy. Príkazom zistiť cestu
``` 
  >>where conda
  C:\Users\mmahe\anaconda3\python.exe
  C:\Users\mmahe\AppData\Local\Microsoft\WindowsApps\python.exe
  >>where python
  C:\Users\mmahe\anaconda3\Library\bin\conda.bat
  C:\Users\mmahe\anaconda3\Scripts\conda.exe
  C:\Users\mmahe\anaconda3\condabin\conda.bat
``` 
Následne ho systémovej premennej PATH pridať obe cesty, v tomto prípade
*C:\Users\mmahe\anaconda3_ a _C:\Users\mmahe\anaconda3\Scripts_. Po inštalácii
overiť, či je nainštalovaná otvorením súboru Anaconda Prompt a napísaním príkazu
_--version_, ktorá vypíše aktuálne verziea zapísať ju do systémových premenných. Klávesovou skratkou windows + R otvoriť
spustenie a napísať príkaz sysdm.cpl.
```
>>conda --version
conda 4.12.0
>>python --version
Python 3.9.12
``` 
6. Na vytvorenie nového prostredia superpozicia s Python verziou 3.9 a jeho aktiváciu
sú potrebné príkazy:
```
>>conda create --name superposition python=3.9
>>conda activate superposition
```
7. Na podporu hlbokého učenia je potrebné nainštalovať PyTorch pomocou príkazu
```
>>conda install pytorch torchvision torchaudio cpuonly -c pytorch
```
pri inštalácii na všetky otázky odpovedať áno.
8. Spustiť inštaláciu prostredia PyCharm a otvoriť ho. Zvoliť možnosť otvoriť projekt
z VCS. Do URL vložiť link https://github.com/xmahely/BP-Superpozicia.git.
A stlačiť tlačidlo Clone. V prostredí ísť do nastavení -> projekt -> python interpreter a pridať nový. Vybrať
Conda environmnent a vybrať existujúce prostredie superpozícia. Kliknúť na ok.
V tomto kroku bude potrebné pridať ďalšie knižnice, ktoré sú použité na implementáciu. Na to jednoducho stačí kliknúť na plusko a vyhľadať a nainštalovať knižnice:
sqlalchemy
- sqlalchemy-utils
- pyodbc
- python-levensthein
- pandas
9. Algoritmus na vyhľadanie a označenie superpozície sa spúšťa cez súbor _main.py_.
