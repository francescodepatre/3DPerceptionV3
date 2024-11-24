<!DOCTYPE html>
<html lang="it">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <h1>Documentazione del Sistema di Rilevamento e Elaborazione Dati</h1>
</head>
<body>
    <h1>Sistema di Rilevamento e Elaborazione Dati</h1>
    <p>Questo repository contiene diversi script per il rilevamento di oggetti, l'elaborazione dei dati acquisiti da sensori (RGB e profondità), e l'analisi multimodale utilizzando un modello LSTM. È suddiviso in tre file principali:</p>
    <ul>
        <li><strong>online_detection.py</strong>: esegue il rilevamento in tempo reale utilizzando YOLOv11 per identificare oggetti nelle immagini RGB.</li>
        <li><strong>offline_detection.py</strong>: esegue il rilevamento su file video pre-registrati (RGB e profondità), applicando YOLOv11 e salvando i risultati.</li>
        <li><strong>post_processing.py</strong>: gestisce i dati ottenuti, calcola le distanze e aggiorna una mappa in tempo reale, salvando i risultati in un file CSV.</li>
    </ul>

    <h2>Requisiti</h2>
    <p>Prima di eseguire gli script, assicurati di avere i seguenti pacchetti Python installati:</p>
    <ul>
        <li><code>opencv-python</code></li>
        <li><code>numpy</code></li>
        <li><code>folium</code></li>
        <li><code>requests</code></li>
        <li><code>torch</code></li>
        <li><code>shapely</code></li>
        <li><code>pyproj</code></li>
        <li><code>pandas</code></li>
        <li><code>ultralytics</code> (per YOLOv11)</li>
        <li><code>torchvision</code></li>
    </ul>
    <p>Per installarli, puoi usare <code>pip</code>:</p>
    <pre><code>pip install opencv-python numpy folium requests torch shapely pyproj pandas ultralytics torchvision</code></pre>

    <h2>Descrizione degli Script</h2>

    <h3>1. online_detection.py</h3>
    <p>Questo script esegue il rilevamento in tempo reale di oggetti (persone) utilizzando YOLOv11 su flussi video da telecamere RGB. Le informazioni relative alle posizioni degli oggetti vengono salvate in tempo reale.</p>
    <p><strong>Funzioni principali:</strong></p>
    <ul>
        <li>Caricamento del modello YOLO.</li>
        <li>Rilevamento di persone nel flusso video.</li>
        <li>Acquisizione delle coordinate delle persone e calcolo della distanza tramite mappa di profondità.</li>
    </ul>

    <h3>2. offline_detection.py</h3>
    <p>Questo script è simile al precedente, ma esegue il rilevamento su file video pre-registrati (RGB e profondità), permettendo di elaborare i dati offline.</p>
    <p><strong>Funzioni principali:</strong></p>
    <ul>
        <li>Caricamento di video RGB e di profondità.</li>
        <li>Applicazione di YOLOv11 per il rilevamento degli oggetti.</li>
        <li>Salvataggio dei risultati delle rilevazioni in un file di log.</li>
    </ul>

    <h3>3. post_processing.py</h3>
    <p>Questo script elabora i risultati ottenuti dai rilevamenti, calcola le distanze reali e predette, e aggiorna una mappa interattiva con le posizioni degli oggetti rilevati.</p>
    <p><strong>Funzioni principali:</strong></p>
    <ul>
        <li>Calcolo delle distanze reali e predette.</li>
        <li>Salvataggio delle distanze in un file CSV.</li>
        <li>Aggiornamento della mappa con le posizioni GPS, utilizzando il modulo <code>folium</code>.</li>
        <li>Creazione di una mappa HTML interattiva con le posizioni degli oggetti rilevati.</li>
    </ul>

    <h2>Utilizzo</h2>

    <h3>1. online_detection.py</h3>
    <p>Esegui lo script per il rilevamento in tempo reale:</p>
    <pre><code>python online_detection.py</code></pre>
    <p>Lo script eseguirà il rilevamento delle persone nel flusso video e salverà i risultati in un file CSV.</p>

    <h3>2. offline_detection.py</h3>
    <p>Per eseguire il rilevamento su un file video esistente, utilizza il seguente comando:</p>
    <pre><code>python offline_detection.py</code></pre>
    <p>Il file video dovrebbe essere posizionato nella stessa directory dello script o specificare il percorso corretto nei parametri.</p>

    <h3>3. post_processing.py</h3>
    <p>Dopo aver eseguito uno dei due script precedenti, utilizza <code>post_processing.py</code> per elaborare i dati raccolti:</p>
    <pre><code>python post_processing.py</code></pre>
    <p>Questo script aggiornerà la mappa e salverà un file CSV con le distanze calcolate.</p>

    <h2>Output</h2>
    <ul>
        <li><strong>Mappe interattive:</strong> Verranno generate mappe in formato HTML con le posizioni degli oggetti rilevati.</li>
        <li><strong>File CSV:</strong> Ogni script salverà un file CSV contenente i dati relativi alle distanze reali e predette.</li>
    </ul>

    <h2>Contribuire</h2>
    <p>Se desideri contribuire a questo progetto, fai un fork del repository e invia una pull request con le modifiche.</p>

    <h2>Licenza</h2>
    <p>Questo progetto è sotto la licenza MIT - vedi il file <a href="LICENSE">LICENSE</a> per dettagli.</p>
</body>
</html>
