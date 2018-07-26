# Felsmalerei_Tiere/ Künstliche Intelligenz
Ziel des Projektes:   - Erkennung von verschiedenen Tierarten der Felsmalerei des Upper Brandberg
                      - Große Tiergruppen, die das Programm erkennen soll: Zebra, Vogel, Antilope, Giraffe (evtl. Ausweitung auf
                      andere Tierarten möglich)
                      
                      
Vorgehen und Schwierigkeiten: 
- Unsere erste Aufgabe war das Raussuchen der Bilder
                        -> Orientierung an der Excel Tabelle
                        -> 1. Problem: unübersichtlich und zu viele Tiere
                        -> Lösung: die großen Tierarten herausgefiltert
                        -> 2. Problem: die gegebenen Bilder waren auch etwas unübersichtlich
                        -> Lösung: Umbenennung der Bilddateien, Dateinamen den Sites, Figures, Plates und Fixes anpassen
                        
- Da alle drei Teilnehmer der Gruppe vorher noch nicht mit Künstlicher Intelligenz, neuralen Netzwerken etc. gearbeitet haben, 
  lag der nächste Schritt darin, uns damit vertraut zu machen. Die Entscheidung des Frameworks fiel auf Tensorflow.
  Wir sind einem Tutorial gefolgt, in welchem mit dem Codelab Tensorflow for Poets gearbeitet wurde. 
                       -> Problem: man brauchte mindestens 200 Bilder pro Kategorie
                       -> Lösung: manuelles Vervielfältigen der Bilder
  Das Programm hat nach mehrerem Rumprobieren und manchen Fehlerbehebungen funktioniert.
  Jedoch: Kein eigen geschriebener Code vorhanden
  
- Nach Absprache mit dem Dozenten haben wir von vorne begonnen und erst einmal erneut über Tensorflow und seine Möglichkeiten recherchiert
- Wieder sind wir mehreren Tutorials gefolgt und haben uns dann dazu entschieden, das Programm mit Hilfe der 
   Python Deep Learning library Keras zu schreiben
- So ist ein Programm entstanden, das ein Deep-Learning-Model erstellt und aufgrund von train- und test-Dateien in entsprechenden Ordnerstrukturen trainiert werden kann
- Nach dem Training lässt sich das Model speichern und somit zu einem späteren Zeitpunkt wieder laden
- Übergibt man dem Programm ein Bild mit einem Felsmalerei-Ausschnitt, der ein Tier zeigt, gibt es nun zurück, mit welcher Wahrscheinlichkeit dieses Tier sich jeweils in die entsprechenden Kategorien einsortieren lässt
- Diese Klassifizierung funktioniert, der keras-eigenen Evaluierung zufolge, mit einer Accuracy von ca. 97% und einem Loss von 0,1. Schwierigkeiten bereitet jedoch vor allem noch die Unterscheidung zwischen Vögeln und Giraffen.
