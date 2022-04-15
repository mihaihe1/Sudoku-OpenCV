# Sudoku-OpenCV

Pentru extragerea careului Sudoku din imagini am definit functia preprocess_image pentru a gasi cel mai mare patrat din imagine(careul sudoku).

Dupa preprocesarea corecta a colturilor careului pentru imaginile de antrenare, urmatorul pas a fost extragerea careului si alinierea acestuia cu axele Ox si Oy. Pentru a realiza asta am utilizat functia cv.getPerspectiveTransform pentru a calcula transformation matrix-ul asociat punctelor ce determina colturile careului. Dupa aceasta am utilizat functia cv.warpPerspective  careia I-a fost trimisa ca parametru matricea calculata anterior si imaginea originala, rezultand o imagine cu dimensiunea 500x500 ce contine numai careul sudoku.  

Rezolvare task 1  

Am extras fiecare patch din careu, folosindu-ma de faptul ca matricea are dimensiunea 500x500 si sunt 9 patch-uri pe fiecare linie/coloana, astfel fiecare patch are o dimensiune de aproximativ 55x55, unde 55 = 500 // 9. Pentru a determina daca in patch se afla sau nu o cifra, am transformat prima data patch-ul intr-unul grayscale, caruia i-am aplicat GaussianBlur si apoi un adaptiveThreshold ce accentueaza informatia din patch. Dupa ce am patch-ul modificat gasesc cel mai mare contur din poza pentru a determina daca exista o cifra in poza sau nu. Am observat dupa mai multe iteratii ca un threshold bun pentru cea mai mare arie este de 50. Astfel, daca aria cea mai mare gasita in patch este mai mica decat 50 inseamna ca patch-ul nu contine o cifra, respectiv contine una daca aria e mai mare decat 50.  

Rezolvare task 2  

Pentru a determina daca avem jigsaw colorat sau nu, am calculat pentru patch-urile de pe prima linie a careului media pe fiecare din cele 3 canale. Am observat ca pentru cazul alb-negru diferenta dintre maximul si minimul mediei celor 3 canale este foarte mica, iar pentru cel color foarte mare. Am calculat pentru fiecare patch de pe prima linie pentru a ma asigura ca nu exista cumva vreun patch determinat incorect.  

Caz jigsaw colorat:   

Voi taia marginile din fiecare patch, calculez media pe cele 3 canale si creez o matrice in care voi stoca culoarea fiecarui patch. Este clar ca in cazul in care canalul ‘B’ contine valoarea medie maxima, patch-ul este albastru. La fel si pentru ‘R’, cu o diferenta foarte mare fata de celelalte valori, iar ultimul caz ramane patch-ului de culoare galbena ‘Y’. Pentru determinarea cifrelor folosesc aceeasi metoda ca la task-ul 1, doar cu un threshold putin mai mic(45). Pentru a determina in ce regiune se afla un patch folosesc matricea calculata anterior si aplic un algoritm de fill ce va ‘umple’ cu acelasi numar toate celule conectate(vecine si cu aceeasi culoare)  

Caz jigsaw alb-negru:  

Abordarea pe care am ales-o a fost aceea de a calcula pentru fiecare patch daca marginile sunt ingrosate sau nu, pentru a determina care zone vecine sunt accesibile. Astfel, am luat primii 5 pixeli din patch si le-am aplicat un filtru grayscale. Am observat ca media cea mai corecta pentru imaginile de antrenare era sub 140 pentru margine ingrosata si peste pentru celulele ce puteau fi accesate. Pentru a retine aceste informatii am folosit 4 matrici: top_blocked, bottom_blocked, left_blocked si right_blocked. Pentru numerotarea zonelor am folosit tot un algoritm de umplere, folosind informatiile din cele 4 matrici. 
