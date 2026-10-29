//+------------------------------------------------------------------+
//| ExportCalendar.mq5 — export du calendrier économique MT5 en CSV  |
//| Usage : copier dans <dossier de données MT5>\MQL5\Scripts,       |
//|         compiler (F7 dans MetaEditor), puis glisser le script    |
//|         sur n'importe quel graphique.                            |
//| Sortie : <dossier de données MT5>\MQL5\Files\calendar_events.csv |
//| Heures : heure du serveur de trading (comme les bougies MT5).    |
//| Lecture seule : aucun ordre n'est passé.                         |
//+------------------------------------------------------------------+
#property script_show_inputs
#property strict

input datetime InpFrom       = D'2015.01.01 00:00';
input string   InpCurrencies = "USD,EUR,GBP";
input int      InpMinImportance = 1;   // 0 = aucune, 1 = faible, 2 = moyenne, 3 = forte

string Num(const bool has, const double v)
  {
   return has ? DoubleToString(v, 6) : "";
  }

void OnStart()
  {
   int h = FileOpen("calendar_events.csv", FILE_WRITE | FILE_CSV | FILE_ANSI, ',');
   if(h == INVALID_HANDLE)
     {
      Print("Impossible de créer le fichier : ", GetLastError());
      return;
     }
   FileWrite(h, "time_server", "currency", "country_id", "event_id", "event_name", "importance", "frequency",
             "sector", "unit", "actual", "forecast", "previous", "revised_previous", "impact_type");
   string cur[];
   int nc = StringSplit(InpCurrencies, ',', cur);
   long total = 0;
   for(int c = 0; c < nc; c++)
     {
      // fenêtres annuelles pour limiter le volume de chaque requête
      for(datetime t0 = InpFrom; t0 < TimeCurrent(); t0 += 365 * 86400)
        {
         datetime t1 = MathMin(t0 + 365 * 86400, TimeCurrent());
         MqlCalendarValue vals[];
         int n = CalendarValueHistory(vals, t0, t1, NULL, cur[c]);
         if(n <= 0)
            continue;
         for(int i = 0; i < n; i++)
           {
            MqlCalendarEvent ev;
            if(!CalendarEventById(vals[i].event_id, ev))
               continue;
            if((int)ev.importance < InpMinImportance)
               continue;
            string name = ev.name;
            StringReplace(name, ",", " ");
            FileWrite(h, TimeToString(vals[i].time, TIME_DATE | TIME_SECONDS), cur[c], (string)ev.country_id,
                      (string)vals[i].event_id, name, (int)ev.importance, (int)ev.frequency, (int)ev.sector, (int)ev.unit,
                      Num(vals[i].HasActualValue(), vals[i].GetActualValue()),
                      Num(vals[i].HasForecastValue(), vals[i].GetForecastValue()),
                      Num(vals[i].HasPreviousValue(), vals[i].GetPreviousValue()),
                      Num(vals[i].HasRevisedValue(), vals[i].GetRevisedValue()),
                      (int)vals[i].impact_type);
            total++;
           }
        }
     }
   FileClose(h);
   PrintFormat("Export terminé : %I64d publications dans MQL5\\Files\\calendar_events.csv", total);
  }
//+------------------------------------------------------------------+
