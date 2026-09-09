# EyeSee
Questo progetto ha lo scopo di comparare vari modelli di classificazione di immagini per diagnosticare varie malattie oculari(retinopatia diabetica, glaucoma, cataratte o se è sano) da una foto della retina.<br/><br/>
Tra i modelli utilizzati figura uno di nostra creazione, nominato EyeSeeNet, ed altri tre di progressiva grandezza, più complessi del primo citato: MobileNetV2, EfficientNetB0 e ResNet50V2.<br/><br/>
Di questi ultimi analizzeremo le performance sia allenando unicamente il nuovo livello denso aggiunto, sia scongelando in aggiunta l'ultimo blocco convolutivo, per poter evidenziare la differenza in performance resa possibile dal fine-tuning.
## Dataset
Il dataset utilizzato è il seguente: https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification <br/><br/>
A causa della grandezza del dataset l'originale non è presente, bensì vi sarà solamente la versione ottenuta grazie al notebook di preprocessing.
## Notebooks
Il progetto comprende 4 notebooks che vanno eseguiti in questo ordine:
### preproc_EyeSee
Serve per trasformare le immagini originali alla risoluzione scelta dal progetto (224x224) e a compiere lo split del dataset secondo una ripartizione 70/15/15 - rispettivamente train, validation e test.<br/><br/>
Dopo l'operazione di split il dataset ottenuto viene organizzato su Drive nelle cartelle corrispondenti, a seconda della partizione (es. test) e della classe a cui appartiene l'immagine (es. glaucoma).
### training_EyeSee
È dove i modelli vengono creati e allenati.<br/><br/>
Il notebook è stato costruito per resistere agli eventuali stacchi del runtime di Google Colab e ciò lo rende più caotico dello stretto necessario, ma si è dimostrata una necessità dopo varie perdite di ore di esecuzione; il notebook salva spesso i modelli generati all'interno della cartella `results/models` e presenta varie celle in cui si può caricare il modello prima di continuare l'addestramento.<br/><br/>
Nella sezione chiamata Ambiente si fà il collegamento con la cartella Drive, si caricano le librerie necessarie, il train/val split e soprattutto viene definito il `MODEL_CONFIG`, dizionario indispensabile per la creazione dei modelli pre-allenati, che consente di caricare il modello da TensorFlow, caricare il preprocessing dell'input corretto e selezionare in corrispondenza di quale blocco scongelare i pesi per il finetuning.<br/><br/>
In caso si voglia allenare un nuovo modello non ancora presente, bisognerà modificare questo dizionario aggiungendovi le rispettive informazioni; inoltre, per semplificare la selezione del nome del blocco è stata introdotta una cella di codice commentata che stampa i nomi dei layer.<br/><br/>
Sucessivamente, nella sezione Funzioni sono presenti le varie funzioni utilizzate dai modelli (con l'eccezione di EyeSeeNet che adopera funzioni leggermente modificate) per essere costruiti, trovare gli iperparametri, essere allenati, parzialmente scongelati ed infine una versione modificata della ricerca di parametri e di allenamento.<br/><br/>
Nella costruzione dei modelli si è deciso di non applicare la data augmentation, che solitamente viene applicata, poichè il dataset è composto da immagini ottenibili esclusivamente tramite macchinari specializzati e ritraenti ambedue gli occhi; dunque, cambiamenti di rotazione o specularità non sono stati ritenuti utili, mentre altri, come la variazione dei colori e della luminosità, si potrebbero rivelare informazioni cruciali per la classificazione.<br/><br/>
I modelli all'interno sono stati allenati finchè non hanno iniziato a mostrare forti segni di overfitting, perciò non tutte le celle sono state eseguite.
### test_EyeSee
I modelli allenati vengono caricati in sequenza calcolando la accuracy, l'F1 score e la confusion matrix, sia della versione con il solo nuovo dense layer che della versione fine-tuned. Infine le metriche calcolate sono messe in un dataframe Pandas, salvate in un CSV e mostrate in un barchart; tutti questi risultati vengono salvati all'interno di `results/graphs`.
### gradCAM_EyeSee
In conclusione, i modelli fine-tuned vengono caricati in questo notebook, e grazie a gradCAM riusciamo ad osservare il "ragionamento" dietro le decisioni dei modelli.<br/><br/>
A questo scopo sono state scelte manualmente 4 campioni, tutti appartenenti a classi distinte, per permettere di rilevare le zone motivanti la previsione di ciascuna patologia.<br/><br/>
Oltre alla funzione `loadImage` che carica le immagini restituendole direttamente come array, sono presenti due funzioni fondamentali: `GenHeatmap`, che genera una "mappa di calore" con valori compresi tra 0 e 1 rappresentanti l'influenza decisiva delle diverse aree dell'immagine; `OverlayColormap`, il cui fine è mappare la palette di colori alla heatmap e in seguito applicare quest'ultima in sovrapposizione all'immagine originale.<br/><br/>
In particolare, `GenHeatmap` compie un unico forward pass per ricavare l'output (tensore delle attivazioni neurali) dell'ultimo livello convolutivo precedente al ridimensionamento (pooling) e l'output del modello intero, ovvero le probabilità.<br/><br/>
`GradientTape` permette di tracciare i gradienti calcolati, qualora si volesse risalire a una derivata intermedia come nel nostro caso; infatti avremo bisogno della derivata dello score di classe maggiore rispetto alle attivazioni dell'ultimo layer convolutivo, precisamente: di quanto varia la probabilità alla minima variazione di attivazione di ciascun neurone?<br/><br/>
In prosieguo, scartiamo le informazioni di tutte le dimensioni eccetto quella relativa ai canali - che teoricamente rilevano una feature ciascuno - eseguendo la media, rappresentando un vettore di pesi di importanza di questi ultimi.<br/><br/>
Occorre puntualizzare che nella riga di codice
```
heatmap = last_layer_activations[0] @ W_channels[..., tf.newaxis]
```
`last_layer_activations[0]` è un tensore con 3 dimensioni (`width`, `height`, `channels`), mentre `W_channels` è vettore monodimensionale convertito in matrice 2D per consentire l'operazione; pertanto `matmul`, come riportato nella documentazione ufficiale di TensorFlow, tratta i tensori di rango maggiore di 2 come pile di matrici residenti nelle ultime 2 dimensioni (dette "interne"), mentre le restanti dimensioni (dette "esterne") vengono gestite secondo le regole di broadcasting (es. (7, 7, 1280) @ (1280, 1) sarà (7, 1280) @ (1280, 1) per ognuna delle 7 matrici - come indicato dalla dimensione esterna).<br/><br/>
Il risultato passa per `tf.squeeze`, perdendo la dimensione aggiuntiva artefatto di `matmul`, e per `tf.maximum`, applicando una ReLU a tutti gli effetti, mantenendo solo i "pixel" della griglia con impatto positivo sulla classe target, scartando invece quelli con impatto negativo - rendendo la heatmap intuitiva.<br/><br/>
Per finire, `OverlayColormap` si occupa di mappare la palette di colori alla heatmap e di sovrapporla sull'immagine originale.
## Licenza
Il database utilizzato viene distribuito sotto licenza Open Database v1.0.