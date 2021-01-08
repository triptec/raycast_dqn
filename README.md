`npm i triptec/chart-stream -g`
`(head -n1 && tail -f -n 1000) < 2021-01-08T17:25:05.544832451+00:00.csv | stdbuf -oL cut -d "," -f2,12,13,16 | chart-stream`

`(head -n1 && tail -f -n 20000) < 2021-01-08T16:13:35.493729229+00:00.csv | stdbuf -oL awk 'NR == 1 || NR % 10 == 0' | stdbuf -oL cut -d "," -f2,12,13,16 | chart-stream`
