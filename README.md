# Felsmalerei_Tiere/ Künstliche Intelligenz
Ziel des Projektes:   - Erkennung von verschiedenen Tierarten der Felsmalerei des Upper Brandberg
                      - Große Tiergruppen, die das Programm erkennen soll: Zebra, Bird, Antilope, Giraffe (evtl. Ausweitung auf
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
  
- Jetziger Stand: Ein Programm existiert, muss jedoch noch an unsere Bedinungen und unser Ziel angepasst werden: 
Fehlermeldungen beseitigen

