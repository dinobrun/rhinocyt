(deftemplate cellula
        (slot nome (type SYMBOL))
        (slot grado (type INTEGER))
)


(deftemplate diagnosi
        (slot nome (type STRING))
		(multislot informazioni (type STRING))
)

(deftemplate prick-test
        (slot esito (type SYMBOL) (default sconosciuto))
		(slot periodo (type SYMBOL) (default sconosciuto))
		(slot allergene (type SYMBOL) (default sconosciuto))
)

(deftemplate famiglia
		(slot soggetto (type SYMBOL) (allowed-values genitore fratello))
		(slot disturbo (type SYMBOL) (allowed-values allergia poliposi asma))
		(slot tipo (type SYMBOL) (default non-necessario))
)

(deftemplate sintomo
		(slot nome (type STRING))
)

(deftemplate scoperta
		(slot parte-anatomica (type SYMBOL))
		(slot caratteristica (type STRING)  (default "non rilevante"))
)

(deftemplate rinomanometria
		(slot resistenza (type FLOAT))
		(slot caratteristica (type SYMBOL))
)