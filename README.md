# diplomski
ISIC Challange 2020 

documentation
    - Predlog diplomskog rada - pdf
    - Neprostrasna detekcija melanoma - rad iz predmeta Pisana i govorna komunikacija u tehnici

train.py - skripta za treniranje modela

predict.py - skripta za predikciju 

models
    - efficientNet.py - efficient net B0 koriscen za treniranje modela
    - loss.py - custom loss funkcija koja koristi BinaryCrossEntropy dodatno ponderiše grešku u zavisnosti od Monk tona kože 

classification
    - augmentation.py - funkcija koriscena za augmentovanje slika klasifikovanih kao melanom
    - classification.py - pomocne funkcije koje su odredjivale boju kože
    - dataset.py - klasa pomoću koje je napravljen dataset
    - duplicates.py - pomocna funkcija koja je otklonila sve duplikate iz dataseta
    - loader.py - klasa koja sluzi za ucitavanje i prolazenje kroz dataset
    - monk.py - pomocna funkcija koja je dodala metapodatak o boji kože u csv fajl sa metapodacima
    - reducing.py - pomocna funkcija koja je izbalansirala dataset po klasama boje kože

make_test_dataset
    makecsv.py - pomocna funkcija koriscena prilikom pravljenja test data seta
