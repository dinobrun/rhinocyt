(defrule start
		(declare (salience 10000))
=>
		(controllo-calcolabilita)
)

(defrule condizione-normale
		(declare (salience 100))
        (cellula (nome Eosinophil) (grado 0))
        (cellula (nome Mastocyte) (grado 0))
        (or (cellula (nome Neutrophil) (grado 0)) (cellula (nome Neutrophil) (grado 1)))
=>
        (assert (diagnosi(nome "condizione normale") (informazioni "Grado di  Eosinophil: 0" "Grado di  Mastocyte: 0" "Grado di  Neutrophil: 0-1")))
)

(defrule riniti-medicamentosa
		(declare (salience 100))
        (cellula (nome Eosinophil) (grado 0))
        (cellula (nome Mastocyte) (grado 0))
        (or (cellula (nome Neutrophil) (grado 0)) (cellula (nome Neutrophil) (grado 1)))
		(or (sintomo(nome ?ostruzione&:(eq (sub-string 1 10 ?ostruzione) "Ostruzione")))
		(sintomo(nome "Uso eccessivo di farmaci")))
		;decongestionanti positivi, stato gravidico
		;sintomi: ostruzione nasale grave (effetto rebound)
=>
        (assert (diagnosi(nome "rinite medicamentosa") (informazioni "Grado di  Eosinophil: 0" "Grado di  Mastocyte: 0" "Grado di  Neutrophil: 0-1" (aggiungi-informazioni (create$ "Ostruzione" "Uso eccessivo di farmaci")))))
)

(defrule rinite-allergica
		(declare (salience 100))
        (cellula (nome Eosinophil) (grado ?gradoE&:(and(> ?gradoE 0) (< ?gradoE 5))))
        (cellula (nome Mastocyte) (grado ?gradoM&:(and(> ?gradoM 0) (< ?gradoM 5))))
        (cellula (nome Neutrophil) (grado ?gradoN&:(and(> ?gradoN 0) (< ?gradoN 5))))
		(prick-test(esito positivo))
		(or (sintomo(nome ?ostruzione&:(eq (sub-string 1 10 ?ostruzione) "Ostruzione"))) 
			(sintomo(nome "Prurito nasale")) 
			(sintomo(nome "Prurito congiuntivale")) 
			(sintomo(nome ?starnutazione&:(eq (sub-string 1 13 ?starnutazione) "Starnutazione"))) 
			(sintomo(nome "Lacrimazione"))
			(sintomo(nome ?rinorrea&:(eq (sub-string 1 8 ?rinorrea) "Rinorrea"))))
		;sintomi: ostruzione nasale, prurito, starnutazione, lacrimazione, prurito congiuntivale, rinorrea
=>
        (assert (diagnosi(nome "rinite allergica") (informazioni (str-cat "Grado di Eosinophil: " ?gradoE) (str-cat "Grado di Mastocyte: " ?gradoM) (str-cat "Grado di Neutrophil: " ?gradoN) (aggiungi-informazioni (create$ "Prurito nasale" "Prurito congiuntivale" "Ostruzione" "Starnutazione" "Rinorrea" "Lacrimazione") "Prick-test positivo"))))
)

(defrule NARES
		(declare (salience 100))
        (cellula (nome Eosinophil) (grado ?gradoE&:(and(> ?gradoE 0) (< ?gradoE 5))))
        (cellula (nome Mastocyte) (grado 0))
        (cellula (nome Neutrophil) (grado ?gradoN&:(and(>= ?gradoN 0) (< ?gradoN 5))))
		(prick-test(esito negativo))
		(or (sintomo(nome ?rinorrea&:(eq (sub-string 1 8 ?rinorrea) "Rinorrea")))
			(sintomo(nome ?starnutazione&:(eq (sub-string 1 13 ?starnutazione) "Starnutazione"))) 
			(sintomo(nome ?olfatto&:(eq (sub-string 1 8 ?olfatto) "Problemi"))))
		;sintomi: naso chiuso, starnutazione (indipendente), rinorrea (niente prurito)
=>
        (assert (diagnosi(nome "NARES") (informazioni (str-cat "Grado di Eosinophil: " ?gradoE) "Grado di  Mastocyte: 0" (str-cat "Grado di Neutrophil: " ?gradoN) (aggiungi-informazioni (create$ "Rinorrea" "Starnutazione" "Problemi")))))
)

(defrule NARESMA
		(declare (salience 100))
        (cellula (nome Eosinophil) (grado ?gradoE&:(and(> ?gradoE 0) (< ?gradoE 5))))
        (cellula (nome Mastocyte) (grado ?gradoM&:(and(> ?gradoM 0) (< ?gradoM 5))))
        (cellula (nome Neutrophil) (grado ?gradoN&:(and(>= ?gradoN 0) (< ?gradoN 5))))
		(or (sintomo(nome ?rinorrea&:(eq (sub-string 1 8 ?rinorrea) "Rinorrea")))
			(sintomo(nome ?starnutazione&:(eq (sub-string 1 13 ?starnutazione) "Starnutazione"))) 
			(sintomo(nome ?olfatto&:(eq (sub-string 1 8 ?olfatto) "Problemi"))))
		;sintomi: naso chiuso, starnutazione (indipendente), rinorrea (niente prurito)
=>
        (assert (diagnosi(nome "NARESMA") (informazioni (str-cat "Grado di Eosinophil: " ?gradoE) (str-cat "Grado di Mastocyte: " ?gradoM) (str-cat "Grado di Neutrophil: " ?gradoN) "Prick-test negativo" (aggiungi-informazioni (create$ "Rinorrea" "Starnutazione" "Problemi")))))
)

(defrule rinite_mastocitaria
		(declare (salience 100))
        (cellula (nome Mastocyte) (grado ?gradoM&:(and(> ?gradoM 0) (< ?gradoM 5))))
		(or (sintomo(nome ?rinorrea&:(eq (sub-string 1 8 ?rinorrea) "Rinorrea")))
			(sintomo(nome ?starnutazione&:(eq (sub-string 1 13 ?starnutazione) "Starnutazione"))) 
			(sintomo(nome ?olfatto&:(eq (sub-string 1 8 ?olfatto) "Problemi"))))
		;sintomi: naso chiuso, starnutazione (indipendente), rinorrea (niente prurito)
=>
        (assert (diagnosi(nome "rinite mastocitaria") (informazioni (str-cat "Grado di Mastocyte: " ?gradoM) (aggiungi-informazioni (create$ "Rinorrea" "Starnutazione" "Problemi")))))
)

(defrule NARNE
		(declare (salience 100))
        (cellula (nome Neutrophil) (grado ?gradoN&:(and(> ?gradoN 0) (< ?gradoN 5))))
		(sintomo(nome ?ostruzione&:(eq (sub-string 1 10 ?ostruzione) "Ostruzione")))
		; RINITE NEUTROFILA ostruzione nasale, bruciore nasale
=>
        (assert (diagnosi(nome "NARNE") (informazioni (str-cat "Grado di Neutrophil: " ?gradoN) ?ostruzione)))
)

(defrule riniti-irritativa
		(declare (salience 100))
        (cellula (nome Eosinophil) (grado 0))
        (cellula (nome Mastocyte) (grado 0))
        (cellula (nome Neutrophil) (grado ?gradoN&:(and(> ?gradoN 0) (< ?gradoN 5))))
		(sintomo(nome ?ostruzione&:(eq (sub-string 1 10 ?ostruzione) "Ostruzione")))
		;come la neutrofila
		;atrofica: ostruzione nasale, epistassi, presenza di croste nasali (mucosa secca)
=>
        (assert (diagnosi(nome "rinite irritativa") (informazioni "Grado di  Eosinophil: 0" "Grado di  Mastocyte: 0" (str-cat "Grado di Neutrophil: " ?gradoN) ?ostruzione)))
)

(defrule rinosinusite
		(declare (salience 100))
        (cellula (nome Eosinophil) (grado 0))
        (cellula (nome Mastocyte) (grado 0))
        (cellula (nome Neutrophil) (grado ?gradoN&:(and(> ?gradoN 0) (< ?gradoN 5))))
		(or (sintomo(nome ?ostruzione&:(eq (sub-string 1 10 ?ostruzione) "Ostruzione"))) 
			(sintomo(nome "Prurito nasale")) 
			(sintomo(nome ?rinorrea&:(eq (sub-string 1 8 ?rinorrea) "Rinorrea")))
			(sintomo(nome "Febbre")))
		;algie facciali, febbre, dolore gravativo
=>
        (assert (diagnosi(nome "rinosinusite") (informazioni "Grado di  Eosinophil: 0" "Grado di  Mastocyte: 0" (str-cat "Grado di Neutrophil: " ?gradoN) (aggiungi-informazioni (create$ "Prurito nasale" "Ostruzione" "Rinorrea" "Febbre")))))
)

(defrule rinite_micotica
		(declare (salience 100))
        (cellula (nome Eosinophil) (grado 0))
        (cellula (nome Mastocyte) (grado 0))
        (cellula (nome Neutrophil) (grado ?gradoN&:(and(> ?gradoN 0) (< ?gradoN 5))))
		(or (sintomo(nome ?ostruzione&:(eq (sub-string 1 10 ?ostruzione) "Ostruzione"))) 
			(sintomo(nome ?rinorrea&:(eq (sub-string 1 8 ?rinorrea) "Rinorrea"))))
=>
        (assert (diagnosi(nome "rinite micotica") (informazioni "Grado di  Eosinophil: 0" "Grado di  Mastocyte: 0" (str-cat "Grado di Neutrophil: " ?gradoN) (aggiungi-informazioni (create$ "Ostruzione" "Rinorrea")))))
)

(defrule poliposi-nasale-ereditata
		(declare (salience 100))
        (cellula (nome Eosinophil) (grado ?gradoE&:(and(> ?gradoE 0) (< ?gradoE 5))))
        (cellula (nome Mastocyte) (grado ?gradoM&:(and(>= ?gradoM 0) (< ?gradoM 5))))
        (cellula (nome Neutrophil) (grado ?gradoN&:(and(>= ?gradoN 0) (< ?gradoN 5))))
		(famiglia (soggetto ?s)(disturbo poliposi)) 
=>
        (assert (diagnosi(nome "poliposi nasale ereditata") (informazioni (str-cat "Grado di Eosinophil: " ?gradoE) (str-cat "Grado di Mastocyte: " ?gradoM) (str-cat "Grado di Neutrophil: " ?gradoN) (str-cat "Un " ?s " ha presentato la poliposi"))))
)

(defrule poliposi-nasale
		(declare (salience 100))
        (cellula (nome Eosinophil) (grado ?gradoE&:(and(> ?gradoE 0) (< ?gradoE 5))))
        (cellula (nome Mastocyte) (grado ?gradoM&:(and(>= ?gradoM 0) (< ?gradoM 5))))
        (cellula (nome Neutrophil) (grado ?gradoN&:(and(>= ?gradoN 0) (< ?gradoN 5))))
		(scoperta (parte-anatomica essudato))
		(or (sintomo(nome ?rinorrea&:(eq (sub-string 1 8 ?rinorrea) "Rinorrea")))
			(sintomo(nome ?ostruzione&:(eq (sub-string 1 10 ?ostruzione) "Ostruzione"))) 			
			(sintomo(nome ?olfatto&:(eq (sub-string 1 8 ?olfatto) "Problemi"))))
		;disturbi olfattivi (anosmia, iposmia), essudato, rinorrea catarrale/purulenta, disturbi respiratori del sonno
=>
        (assert (diagnosi(nome "poliposi nasale") (informazioni "Grado di Eosinophil: 0" "Grado di Mastocyte: 0" (str-cat "Grado di Neutrophil: " ?gradoN) "Essudato" (aggiungi-informazioni (create$ "Rinorrea" "Ostruzione" "Problemi")))))
)

(defrule polipo_antrocoanale
		(declare (salience 100))
        (cellula (nome Eosinophil) (grado 0))
        (cellula (nome Mastocyte) (grado 0))
        (cellula (nome Neutrophil) (grado ?gradoN&:(and(> ?gradoN 0) (< ?gradoN 5))))
		(sintomo(nome ?ostruzione&:(eq (sub-string 1 10 ?ostruzione) "Ostruzione")))
=>
        (assert (diagnosi(nome "poliposi antrocoanale") (informazioni "Grado di Eosinophil: 0" "Grado di Mastocyte: 0" (str-cat "Grado di Neutrophil: " ?gradoN) ?ostruzione)))
)

(defrule citologia-anamnesi-assenti-pricktestpositivo
		(prick-test (esito positivo))
		(test (= (get-number-of-facts-by-name sintomo) 0))
		(test (= (get-number-of-facts-by-name famiglia) 0))
		(test (= (get-number-of-facts-by-name scoperta) 0))
		(test (= (get-number-of-facts-by-name rinomanometria) 0))
		(test (= (get-number-of-facts-by-name cellula) 0))
=>
		(assert (diagnosi (nome "rinite allergica") (informazioni "Sintomatologia assente" "Anamnesi familiare assente" "Esame del medico assente" "Citologia assente" "Prick-test: positivo")))
)

(defrule citologia-anamnesi-assenti-pricktestnegativo
		(prick-test (esito negativo))
		(test (= (get-number-of-facts-by-name sintomo) 0))
		(test (= (get-number-of-facts-by-name famiglia) 0))
		(test (= (get-number-of-facts-by-name scoperta) 0))
		(test (= (get-number-of-facts-by-name rinomanometria) 0))
		(test (= (get-number-of-facts-by-name cellula) 0))
=>
		(assert (diagnosi (nome "non calcolabile") (informazioni "Sintomatologia assente" "Anamnesi familiare assente" "Esame del medico assente" "Citologia assente" "Prick-test: negativo")))
)

(defrule anamnesi-assente-pricktestpositivo
		(or(prick-test (esito positivo) (periodo pollinico) (allergene stagionale)) (prick-test (esito positivo) (allergene perenne)))
		(test (= (get-number-of-facts-by-name sintomo) 0))
		(test (= (get-number-of-facts-by-name famiglia) 0))
		(test (= (get-number-of-facts-by-name scoperta) 0))
		(test (= (get-number-of-facts-by-name rinomanometria) 0))
		(test (= 0 (length$ (find-fact((?d diagnosi)) (eq ?d:nome "condizione normale")))))
=>
		(assert (diagnosi (nome "rinite allergica") (informazioni "Sintomatologia assente" "Anamnesi familiare assente" "Esame del medico assente" "Citologia: negativa" "Prick-test: positivo")))
)

(defrule anamnesi-assente-pricktestpositivo-apollinico
		(prick-test (esito positivo) (periodo apollinico) (allergene stagionale))
		(test (= (get-number-of-facts-by-name sintomo) 0))
		(test (= (get-number-of-facts-by-name famiglia) 0))
		(test (= (get-number-of-facts-by-name scoperta) 0))
		(test (= (get-number-of-facts-by-name rinomanometria) 0))
=>
		(assert (diagnosi (nome "rinite allergica") (informazioni "Sintomatologia assente" "Anamnesi familiare assente" "Esame del medico assente" "Prick-test: positivo con periodo apollinico e allergene stagionale")))
)

(defrule anamnesi-assente-pricktestpositivo-apollinico
		(prick-test (esito positivo) (periodo apollinico) (allergene stagionale))
		(diagnosi (nome "condizione normale"))
		(test (= (get-number-of-facts-by-name sintomo) 0))
		(test (= (get-number-of-facts-by-name famiglia) 0))
		(test (= (get-number-of-facts-by-name scoperta) 0))
		(test (= (get-number-of-facts-by-name rinomanometria) 0))
=>
		(assert (diagnosi (nome "rinite vasomotoria con sensibilizzazione allergenica") (informazioni "Sintomatologia assente" "Anamnesi familiare assente" "Esame del medico assente" "Citologia: negativa" "Prick-test: positivo con periodo apollinico e allergene stagionale")))
)

(defrule citologia-assente
		(prick-test (esito positivo))
		(test(= (length (find-all-facts ((?fct cellula)) TRUE)) 0))
=>
		(assert (diagnosi (nome "rinite allergica") (informazioni "Citologia assente" "Prick-test: positivo")))
)

(defrule rv-allergenica
		(prick-test (esito positivo) (allergene perenne))
		(diagnosi (nome "condizione normale"));CITOLOGIA NEGATIVA: fare la citologia (ossia cacthare le regole con solo citologia) i risultati li metti in un multislot di diagnosi
=>
		(assert (diagnosi (nome "rinite vasomotoria con sensibilizzazione allergenica") (informazioni "Prick-test: positivo, con allergene perenne" "Citologia: negativa")))
)

(defrule rv-allergenica
		(prick-test (esito positivo) (periodo apollinico) (allergene stagionale))
=>
		(assert (diagnosi (nome "rinite vasomotoria con sensibilizzazione allergenica") (informazioni "Prick-test: positivo, con periodo apollinico e allergene stagionale")))
)
