def extract_parts_judgments(f): #extract text from different parts
    import glob,re, os, sys, random

    nn = ['section', 'case_name', 'app_no', 'is_judgement', 'strasbourg' 'date', 'final', 'intro' 'procedure', 'circumstances', 'domestic_law', 'law', 'decision', 'end', 'opinions' ]
        
        
    f = [i for i in f if i != '\n']
    print('Extracting facts', len(f))
    #print(f[:10], '\n')
    section, case_name, app_no, date, final_date, procedure, circumstances = False, False, False, False, 'N/A\n', False, False
    
    ###section, grand chamber###
    for num in range(10):
        section_index = 0
        line = f[num]
        if re.search('COURT', line) or re.search('SECTION', line) or re.search('GRAND CHAMBER', line):
            section = line
            section_index = num
            break
    if section:
        f = f[section_index+1:]
            
    ###case name###
    mistake_line_break = 0
    for num in range(5):
        case_name_index = 0
        line = f[num]
        if re.search('CASES? OF', line) != None:
            case_name = line
            case_name_index = num
            m = re.search('(/*)(\(Application.*)', line)
            if m!=None:
                case_name = m.group(1)
                app_no = m.group(2)
                app_no_index = -1
                mistake_line_break = 1
    if case_name:
        f = f[case_name_index+1:]
    
            
    ###application number###
    if mistake_line_break == 0:
        if re.search('[0-9/]+', f[0]):
            app_no = f[0]
            app_no_index = 0
        else:
            app_no = f[1]
            app_no_index = 1
    
    if app_no:
        f = f[app_no_index+1:]
    
    front_page_end_index = 0
    ###date###
    for num in range(10):
        line = f[num]
        if re.search('STRASBOURG', line):
            date_index = num+1
            date = f[date_index]
            front_page_end_index = date_index
    
    ###final###
    for num in range(len(f)):
        if re.search('FINAL', f[num]):
            final_date = f[num+1]
            front_page_end_index = date_index+2
            break
            
    
    #if final_date == 'N/A\n':
    f = f[front_page_end_index+1:]
        
    #procedure keyword#
    found_procedure = 0
    for num in range(len(f)):
        if re.search('PROCEDURE', f[num]) or re.search('^procedure\n', f[num]):
            procedure_index = num
            found_procedure = 1
            break
    
    
            
    #circumstances keyword
    circumstances_index = False
    found_circumstances = False
    found_facts = False
    for num in range(len(f)):
        if re.search('CIRCUMSTANCES', f[num]): ###SOMETIMES ONLY FACTS
            circumstances_index = num
            found_circumstances = True
            break
        else:
            if re.search('FACTS', f[num]): ###SOMETIMES ONLY FACTS
                circumstances_index = num
                found_facts = True
            else:
                found_circumstances = False
                
                
    ###intro###
    if found_procedure == 1:
        intro = f[:procedure_index]
        f = f[procedure_index:]
    else:
        if found_facts == True:
            intro = f[:circumstances_index-1]
            f = f[circumstances_index-1:]
        else:
            intro = f[:circumstances_index]
            f = f[circumstances_index:]
        
    #circumstances keyword
    circumstances_index = False
    found_circumstances = False
    found_facts = False
    for num in range(len(f)):
        if re.search('CIRCUMSTANCES', f[num]) or re.search('The circumstances of the case', f[num]): ###SOMETIMES ANLY FACTS
            circumstances_index = num
            found_circumstances = True
            break
        else:
            if re.search('FACTS', f[num]): ###SOMETIMES ONLY FACTS
                circumstances_index = num
                found_facts = True
            else:
                found_circumstances = False 

    if found_facts == True:
        procedure = f[:circumstances_index-1]
        f = f[circumstances_index-1:]
    else:
        procedure = f[:circumstances_index]
        f = f[circumstances_index:]
        
        
    #domestic law keyword
    found_domestic =0
    domestic_law_index = 0
    for num in range(len(f)):
        if re.search('DOMESTIC LAW', f[num]) or re.search('(R|r)elevant domestic law', f[num]):
            domestic_law_index = num
            found_domestic = 1
            break
    
    
    
    #THE LAW keyword
    law_index = 0
    for num in range(len(f)):
        if re.search('THE LAW', f[num]) or re.search('as to the law\n', f[num].lower()):
            law_index = num
            #print(law_index)
            
    ###circumstances###
    if found_domestic == 0:
        print('no domestic')
        circumstances = f[:law_index]
        f = f[law_index:]
    if found_domestic == 1:
        circumstances = f[:domestic_law_index]
        f = f[domestic_law_index:]
    
     #THE LAW keyword again
    law_index = 0
    for num in range(len(f)):
        if re.search('THE LAW', f[num]) or re.search('as to the law\n', f[num].lower()):
            law_index = num
            break
            
    ###relevant domestic law###
    if found_domestic == 1:
        domestic_law = f[:law_index]
        f = f[law_index:]
    else:
        domestic_law = []
    
    
    #decision keyword
    verdict_index = 0
    found_verdict = 0
    for num in range(len(f)):
        if re.search('FOR THES?E? REASONS,? THE COURT?', f[num].upper()) or re.search('(F|f)or these reasons,? the (C|c)ourt', f[num]) or re.search('For these reasons,', f[num]):
            verdict_index = num
            found_verdict = 1
            break
    
    if found_verdict == 0:
        print('Verdict not found\n')
        #print(f)
        
    ###THE LAW###
    law = f[:verdict_index]
    f = f[verdict_index:]        
    
    #end keyword
    found_registrar = 0
    end_index = 0
    for num in range(len(f)):
        if re.search('Registrar', f[num]):# or re.search('President', f[num]):
            found_registrar = 1
            end_index = num
            break
    
    if found_registrar == 0:
        for num in range(len(f)):
            if re.search('President', f[num]):# or re.search('President', f[num]):
                found_registrar = 1
                end_index = num
    if found_registrar == 0:
        for num in range(len(f)):
            if re.search('OPINION', f[num]):
                found_registrar = 1
                end_index = num-1

                
        
     ###THE VERDICT###
    verdict = f[:end_index+1]
    f = f[end_index+1:]
    
    #THE OPINION keyword
    num_opinion = []
    for num in range(len(f)):
        if re.search('OPINION', f[num]):
            num_opinion.append(num)
            #print(num)
    
    ##add annex
    annex_index = False
    for num in range(len(f)):
        if re.search('ANNEX', f[num]):
            annex_index = num
    
    
    if annex_index != False:
        annex = f[annex_index:]
        f = f[:annex_index]
    else:
        annex = False
    
    
    ###opinions###
    all_opinions = []
    if num_opinion != []:
        if len(num_opinion) == 1:
            f = f[num_opinion[0]:]
            all_opinions.append(f)
        else:
            for n in range(len(num_opinion)-1):
                all_opinions.append(f[num_opinion[n]:num_opinion[n+1]-1])
            all_opinions.append(f[num_opinion[-1]:])

    #print(all_opinions)        
    return [section, case_name, app_no, date, final_date, intro, procedure, circumstances, domestic_law, law, verdict, all_opinions, annex]
    
            #procedure_index = num
    