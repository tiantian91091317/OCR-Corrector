from FASPell.char_sim import CharFuncs
from FASPell.masked_lm import MaskedLM
from FASPell.bert_modified import modeling
import re
import json
import pickle
import argparse
import numpy
import logging
import FASPell.plot as plot
import tqdm
import time

####################################################################################################

__author__ = 'Yuzhong Hong <hongyuzhong@qiyi.com / eugene.h.git@gmail.com>'
__date__ = '10/09/2019'
__description__ = 'The main script for FASPell - Fast, Adaptable, Simple, Powerful Chinese Spell Checker'


CONFIGS = json.loads(open('FASPell/faspell_configs.json', 'r', encoding='utf-8').read())

WEIGHTS = (CONFIGS["general_configs"]["weights"]["visual"], CONFIGS["general_configs"]["weights"]["phonological"], 0.0)

CHAR = CharFuncs(CONFIGS["general_configs"]["char_meta"])

traditional_char_regular = r'[萬與醜專業叢東絲丟丟兩嚴並喪個爿豐臨為麗舉麼義烏樂喬習鄉書買亂乾亂爭於虧雲亙亙亞亞產畝親褻嚲億僅僕從侖倉儀們價伕眾優夥會傴傘偉傳傷倀倫傖偽佇佇佈體佔佘餘傭僉併來侖俠侶僥偵側僑儈儕儂侶侷俁係俠俁儔儼倆儷儉倀倆倉個們倖倣倫債傾偉傯側偵偸偺僂偽僨償傑傖傘備傚傢儻儐儲儺傭傯傳傴債傷傾僂僅僉僑僕僞僥僨僱價儀儂億儅儈儉儐儔儕儘償優儲儷儸儹儺儻儼兒兇兌兌兒兗兗黨內兩蘭關興茲養獸囅內岡冊冊寫軍農塚冪馮沖決況凍淨凃淒涼淩凍減湊凜凜凟幾鳳処鳧憑凱凱凴擊氹鑿芻劃劉則剛創刪別刪別剗剄劊劌剴劑剄則剋剮劍剗剛剝剝劇剮剴創剷劃劄劇劉劊劌劍劑劒勸辦務勱動勵勁勞勢勁勳動勗務勩勛勝勞勢勣勦勩勱勳勵勸勻勻匭匭匱匯匱匲匳匵區醫區華協協單賣蔔盧鹵臥衛卻卹巹卻廠廳曆厲壓厭厙廁厙厛厠廂厴厤廈廚廄厭廝厰厲厴縣參參靉靆雙發變敘疊叢葉號歎嘰籲後吒嚇呂嗎唚噸聽啟吳吳呂嘸囈嘔嚦唄員咼嗆嗚詠哢嚨嚀噝吒諮噅鹹咼響啞噠嘵嗶噦嘩噲嚌噥喲員哢唄唕唚嘜嗊嘮啢嗩唕喚唸問啓啗啞啟啢啣嘖嗇囀齧囉嘽嘯喒喚喦喪喫喬單喲噴嘍嚳嗆嗇嗊嗎嗚嗩囁噯嗶嘆嘍嘔嘖嘗噓嘜嚶嘩嘮嘯嘰囑嘵嘸嘽噁噅噓嚕噝噠噥噦噯噲噴噸噹嚀嚇嚌嚐嚕嚙囂嚥嚦嚨嚮謔嚲嚳嚴嚶嚻囀囁囂囅囈囉囌囑囘團囪囬園囯囪圍圇國圖圓圇國圍園圓圖團聖壙場壞塊堅壇壢壩塢墳墜坰壟壟壚壘墾坰堊墊埡墶壋塏堖垵垻塒塤堝墊埡垵埰執堅堊塹墮堖堝堦堯報場堿壪塊塋塏塒塗塚塢塤塲塵塹塼墊牆墜墝墮墰墳墶墻墾壇壋壎壓壘壙壚壜壞壟壠壢壩壪壯壯聲殼壺壼壺壼壽夀處備複夠夠夢夥頭誇夾奪夾奩奐奮奐獎奧奧奩奪奮妝婦媽妝嫵嫗媯姍姍薑姦姪婁婭嬈嬌孌娛娬娛媧嫻婁婦婬婭嫿嬰嬋嬸媧媼媮媯媼媽嫋嬡嬪嫗嬙嫵嫺嫻嫿嬀嬈嬋嬌嬙嬝嬡嬤嬪嬭嬰嬤嬸嬾孃孌孫學孿孫學孿寧寶實寵審憲宮宮寬賓寢寢實寧審寫寬寵寶對尋導壽尅將將專尋對導爾塵嘗堯尲尷尷屍盡層屭屆屜屆屍屓屜屬屢屢層屨屨屬屭嶼歲豈嶇崗峴嶴嵐島岡嶺嶽崠巋嶨嶧峽嶢嶠崢巒峴島峽嶗崍嶮崍崐崑崗崙崠崢崬嶄崳嵐嵗嶸嶔崳嶁嶁嶄嶇嶔嶗嶠嶢嶧嶨嶮嶴嶸嶺嶼嶽巔巋巒巔巖鞏巰巰巹幣帥師幃帳簾幟帥帶幀師幫幬帳帶幘幗幀冪幃幗幘襆幟幣幫幬幱幹並幹幾廣莊慶廬廡庫應廟龐廢庫廎廁廂廄廈廎廕廚廝廟廠廡廢廣廩廩廬廳廻廼開異棄弔張彌弳彎弳張強彈強彆彈彊彌彎歸當錄彙彜彞彠彠彥彥彫徹彿徑後徑徠從徠禦復徬徴徵徹憶懺憂愾懷態慫憮慪悵愴憐總懟懌恆戀恥懇惡慟懨愷惻惱惲悅悅愨懸慳憫悵悶悽驚惡懼慘懲憊愜慚憚慣惱惲惻湣愛愜慍憤憒愨愴愷愾願慄慇態慍懾慘慙慚慟慣慤慪慫憖慮慳慴慶慼慾憂憊憐憑憒憖憚憤憫憮憲憶懇應懌懍懣懶懍懕懞懟懣懨懲懶懷懸懺懼懽懾戀戇戇戔戲戔戧戰戧戩戩戯戰戲戶戶紮撲扡扡執擴捫掃揚擾撫拋摶摳掄搶護報擔拋擬攏揀擁攔擰撥擇掛摯攣掗撾撻挾撓擋撟掙擠揮撏挾撈損撿換搗捨捫據捲撚掃掄掗掙掛採擄摑擲撣摻摜揀揚換揮摣揹攬撳攙擱摟攪搆搇損搖搗搶攜搾攝攄擺搖擯攤摑摜摟摣摯摳摶摺摻攖撈撏撐撐撓撚撟撣撥撫撲撳攆擷擼攛撻撾撿擁擄擇擊擋擔擕據擻擠擡擣擧擬擯擰擱擲擴擷擺擻擼擾攄攆攏攢攔攖攙攛攜攝攢攣攤攩攪攬攷敵敍敗敘斂敭數敵數敺斂斃齋斕斕鬥斬斬斷斷於旂無舊時曠暘昇曇晝曨顯時晉晉曬曉曄暈暉晝暫暈暉暘暢曖暫暱曄曆曇曉曏曖曠曡曨曬書會朧劄術樸機殺雜權條來楊榪傑東極構樅樞棗櫪梘棖槍楓梟枴櫃檸柵柺檉梔柵標棧櫛櫳棟櫨櫟欄樹棲樣欒棬椏橈楨榿橋樺檜槳樁桿梔梘條梟夢梱檮棶檢欞棄棖棗棟棧棬棲棶槨椏櫝槧欏橢楊楓楨業極樓欖櫬櫚櫸榦榪榮榿槃構槍槓檟檻檳櫧槧槨槳槼樁樂樅樑樓標樞樣橫檣櫻樷樸樹樺橈橋機橢櫫橫橰櫥櫓櫞檁檉簷檜檟檢檣檁檮檯檳檸檻檾櫂櫃櫓櫚櫛櫝櫞櫟櫥櫧櫨櫪櫫櫬櫳櫸櫺櫻欄權欏欑欒欖欞歡歟歐欽歎歐歗歛歟歡歲歷歸殲歿歿殤殘殞殮殘殫殞殯殤殫殮殯殲毆殺殼毀毀轂毆毉畢斃氈毧毬毿毿氂氌氈氊氌氣氫氣氬氫氬氳氳氹氾匯漢汎汙汚汙湯洶決沍沒遝沖溝沒灃漚瀝淪滄渢溈滬濔況濘淚澩瀧瀘濼瀉潑澤涇潔灑洩洶窪浹淺漿澆湞溮濁測澮濟瀏滻渾滸濃潯濜浹塗涇湧涖濤澇淶漣潿渦溳渙滌潤澗漲澀涼澱淒淚淥淨淩淪淵淶淺淵淥漬瀆漸澠漁瀋滲渙減渢渦溫測遊渾湊湞湣湧湯灣濕潰濺漵漊溈準溝溫溮溳溼滄滅滌滎潷滙滾滯灩灄滿瀅濾濫灤濱灘澦滬滯滲滷滸滻滾滿漁漊漚漢漣濫漬漲漵漸漿潁瀠瀟瀲濰潑潔潙潛潛潟潤潯潰瀦潷潿澀澁澂澆澇澗瀾澠澤澦澩澮澱濁濃瀨瀕濔濕濘濛濜濟濤濫濬濰濱濶濺濼濾瀅瀆瀉瀋瀏瀕瀘瀝瀟瀠瀦瀧瀨瀰瀲瀾灃灄灝灑灕灘灝灣灤灧灩滅燈靈災災燦煬爐燉煒熗炤點為煉熾爍爛烴烏燭煙煩燒燁燴燙燼熱烴煥燜燾無煆煆煇煉煒煖煙煢煥煩煬熒熗熱熲熾燁燄燈燉燐燒燙燜營燦燬燭燴燻燼燾燿爍爐爗爛爭愛爲爺爺爾爿牀牆牋牘牐牘犛牴牽犧牽犢犖犛強犢犧狀獷獁猶狀狽麅獮獰獨狹獅獪猙獄猻狹狽獫獵獼猙玀豬貓蝟獻猶猻獁獃獄獅獎獨獪獫獺獮獰獲獵獷獸獺獻獼玀玆璣璵瑒瑪玨瑋環現瑲璽瑉玨琺瓏珮璫琿現琍璡璉瑣琯琺瓊琿瑉瑋瑒瑣瑤瑩瑪瑯瑲瑤璦璿璉瓔璡璣璦璫環璵璽璿瓊瓏瓚瓔瓚甕甌甌甎甕甖產産甦甯電畫暢畝畢畫異畱佘疇當疇疊癤療瘧癘瘍鬁瘡瘋皰癰痙癢瘂痙痠癆瘓癇癡痺瘂癉瘮瘉瘋瘍瘓瘞瘺瘞瘡瘧癟癱瘮瘺瘻癮癭療癆癇癉癒癘癩癟癡癢癬癤癥癧癩癲癬癭癮臒癰癱癲發皚皚皰皺皸皸皺盃盞鹽監蓋盜盤盜盞盡監盤盧盪瞘眎眡眥眥矓眾著睜睏睞瞼睜睞瞞瞘瞞矚瞼矇矓矚矯矯磯矽礬礦碭碼磚硨硯碸砲礪礱礫礎硜硃矽碩硤磽磑礄硜硤硨確硯硶鹼礙磧磣碩碭堿碸镟確碼磐磑滾磚磣磧磯磽礄礎礙礦礪礫礬礮礱禮禕祐祕禰禎禱禍祿稟祿禪禍禎禕禦禪禮禰禱離禿禿稈秈種積稱穢穠稅穭稈稅稜稟穌稭種稱穩穀穌積穎穡穠穡穢穨穩穫穭窮竊竅窯竄窩窺竇窩窪窶窮窯窰窶窺竄竅竇竈竊豎竝競竪競篤筍筆筧箋籠籩筆筍築篳篩簹箏筧籌簽簡箇箋箏籙箠簀篋籜籮簞簫節範築篋簣簍篛篠篤篩籃籬篳簀簍簑籪簞簡簣簫簷簹簽簾籟籃籌籐籙籜籟籠籢籤籥籩籪籬籮籲糴類秈糶糲粵粧糞粬糧粵糝餱糝糞糢糧糰糲糴糶糸糾紀紂約紅紆紇紈紉紋納紐紓純紕紖紗紘紙級紛紜紝紡紥緊紮細紱紲紳紵紹紺紼紿絀終絃組絆絍絎絏結絕絛絝絞絡絢給絨絰統絲絳縶絹綁綃綆綈綉綌綏綑經綜綞綠綢綣綫綬維綯綰綱網綳綴綵綸綹綺綻綽綾綿緄緇緊緋緍緒緓緔緗緘緙線緜緝緞締緡緣緤緦編緩緬緯緱緲練緶緹緻緼縂縈縉縊縋縐縑縕縗縚縛縝縞縟縣縧縫縭縮縯縱縲縴縵縶縷縹總績繃繅繆繈繒織繕繖繙繚繞繡繢繦繩繪繫繭繮繯繰繳繹繼繽繾纇纈纊續纍纏纓纖纘纜糸糾紆紅紂纖紇約級紈纊紀紉緯紜紘純紕紗綱納紝縱綸紛紙紋紡紵紖紐紓線紺絏紱練組紳細織終縐絆紼絀紹繹經紿綁絨結絝繞絰絎繪給絢絳絡絕絞統綆綃絹繡綌綏絛繼綈績緒綾緓續綺緋綽緔緄繩維綿綬繃綢綯綹綣綜綻綰綠綴緇緙緗緘緬纜緹緲緝縕繢緦綞緞緶線緱縋緩締縷編緡緣縉縛縟縝縫縗縞纏縭縊縑繽縹縵縲纓縮繆繅纈繚繕繒韁繾繰繯繳纘缽罌罈罋罌罎罏網羅罰罷罰羆罵罷罸羈羅羆羈羋羥羨羢羥羨義羶習翹翽翬翬翹翺翽耑耡耮耬耬耮聳恥聶聾職聹聯聖聞聵聰聯聰聲聳聵聶職聹聼聽聾肅肅腸膚膁腎腫脹脅膽勝朧腖臚脛膠脅脇脈脈膾髒臍腦膿臠腳脛脣脩脫脫腡臉脹臘醃腎腖膕腡腦腫齶腳腸膩靦膃騰膁膃臏膕膚膠膩膽膾膿臉臍臏臒臘臚臢臟臠臢臥臨臯臺輿與興舉舊舖舘艤艦艙艫艙艢艣艤艦艪艫艱艱豔艷艸艸藝節羋薌蕪蘆芻蓯葦藶莧萇蒼苧蘇檾苧蘋範莖蘢蔦塋煢繭茲荊荊薦薘莢蕘蓽蕎薈薺蕩榮葷滎犖熒蕁藎蓀蔭蕒葒葤藥蒞莊莖蓧莢莧萊蓮蒔萵薟獲蕕瑩鶯蓴華菴菸萇萊蘀蘿螢營縈蕭薩萬萵葉葒著葠葤葦葯蔥葷蕆蕢蔣蔞蒐蒓蒔蒞蒼蓀蓆蓋藍薊蘺蓡蕷鎣驀蓧蓮蓯蓴蓽蔆蔔蔞蔣蔥蔦蔭蔴薔蘞藺藹蕁蕆蕎蕒蕓蕕蕘蕢蕩蕪蕭蘄蘊蕷薈薊薌薑薔薘薟薦薩藪薰薺藍藎蘚藝藥藪藶藹藺蘀蘄蘆蘇蘊蘋蘚蘞蘢蘭蘺蘿虜慮處虛虛虜號虧蟲虯蟣虯雖蝦蠆蝕蟻螞蠶蠔蜆蠱蠣蟶蠻蟄蛺蟯螄蠐蛺蛻蜆蛻蝸蠟蠅蟈蟬蠍蝕蝟蝦蝨蝸螻蠑螿螄螘螞螢蟎螻螿蟄蟈蟎蠨蟣蟬蟯蟲蟶蟻蠅蠆蠍蠐蠑蠔蠟蠣蠨蠱蠶蠻釁衆衊術銜衚衛衝衞補襯袞衹襖嫋褘襪袞襲襏裝襠褌裊裌裏補裝裡褳襝褲襇製複褌褘褸褲褳襤褸褻繈襆襇襍襏襴襖襝襠襤襪襯襲襴覈見覎規覓覔視覘覜覡覥覦親覬覯覰覲覷覺覻覽覿觀見觀覎規覓視覘覽覺覬覡覿覥覦覯覲覷觝觴觸觶觴觶觸訁訂訃計訊訌討訏訐訒訓訕訖託記訛訝訟訢訣訥訦訩訪設許訴訶診註証詁詆詎詐詒詔評詖詗詘詛詞讋詠詡詢詣試詩詫詬詭詮詰話該詳詵詼詾詿誄誅誆誇譽謄誌認誑誒誕誘誚語誠誡誣誤誥誦誨說説誰課誶誹誼調諂諄談諉請諍諏諑諒論諗諛諜諝諞諡諢諤諦諧諫諭諮諱諳諶諷諸諺諼諾謀謁謂謄謅謊謎謐謔謖謗謙謚講謝謠謨謫謬謳謹謾譁譆證譌譎譏譖識譙譚譜譟譫譭譯議譴護譸譽譾讀讁讅變讋讌讎讐讒讓讕讖讚讛讜讞訁計訂訃認譏訐訌討讓訕訖訓議訊記訒講諱謳詎訝訥許訛論訩訟諷設訪訣證詁訶評詛識詗詐訴診詆謅詞詘詔詖譯詒誆誄試詿詩詰詼誠誅詵話誕詬詮詭詢詣諍該詳詫諢詡譸誡誣語誚誤誥誘誨誑說誦誒請諸諏諾讀諑誹課諉諛誰諗調諂諒諄誶談誼謀諶諜謊諫諧謔謁謂諤諭諼讒諮諳諺諦謎諞諝謨讜謖謝謠謗諡謙謐謹謾謫譾謬譚譖譙讕譜譎讞譴譫讖穀谿豈豎豐豔豬豶豶貍貓貝貞貟負財貢貧貨販貪貫責貯貰貲貳貴貶買貸貺費貼貽貿賀賁賂賃賄賅資賈賉賊賍賑賒賓賔賕賙賚賛賜賞賠賡賢賣賤賦賧質賫賬賭賮賴賵賸賺賻購賽賾贄贅贇贈贊贋贍贏贐贓贔贖贗贛贜貝貞負貟貢財責賢敗賬貨質販貪貧貶購貯貫貳賤賁貰貼貴貺貸貿費賀貽賊贄賈賄貲賃賂贓資賅贐賕賑賚賒賦賭齎贖賞賜贔賙賡賠賧賴賵贅賻賺賽賾贗贊贇贈贍贏贛赬赬趙趕趨趕趙趨趲趲躉躍蹌蹠躒跡踐躂蹺蹕躚躋跼踴躊踐踡蹤躓躑踴蹌躡蹣蹕蹟蹠蹣蹤躕蹺躥躂躉躊躋躍躪躑躒躓躕躚躦躡躥躦躪軀躰軀軃車軋軌軍軑軒軔軛軟軤軫軲軸軹軺軻軼軾較輅輇輈載輊輒輓輔輕輛輜輝輞輟輥輦輩輪輬輭輯輳輸輻輼輾輿轀轂轄轅轆轉轍轎轔轟轡轢轤車軋軌軒軑軔轉軛輪軟轟軲軻轤軸軹軼軤軫轢軺輕軾載輊轎輈輇輅較輒輔輛輦輩輝輥輞輬輟輜輳輻輯轀輸轡轅轄輾轆轍轔辭辤辦辯辮辭辮辯農辳邊遼達遷迆過邁運還這進遠違連遲邇逕迴跡迺適選遜遞逕這連邐週進邏遊運過達違遺遙遜遝遞遠遙適遲遷選遺遼邁還邇邊邏邐鄧鄺鄔郵鄒鄴鄰鬱郃郤郟鄶鄭鄆郟郤酈鄖郵鄲鄆鄉鄒鄔鄕鄖鄘鄧鄭鄰鄲鄴鄶鄺酈醞醱酧醯醬釅釃釀醃醖醜醞醫醬醯醱醻醼釀釁釃釅釋釋裡釐釓釔釕釗釘釙針釣釤釦釧釩釬釵釷釹釺鈀鈁鈃鈄鈅鈈鈉鈍鈎鈐鈑鈒鈔鈕鈞鈡鈣鈥鈦鈧鈮鈰鈳鈴鈷鈸鈹鈺鈽鈾鈿鉀鉄钜鉆鉈鉉鉋鉍鉑鉕鉗鉚鉛鉞鉢鉤鉦鉬鉭鉲鑒鉶鉸鉺鉻鉿銀銃銅銍銑銓銕銖銘銚銛銜銠銣銥銦銨銩銪銫銬鑾銱銲銳銷銹銻銼鋁鋂鋃鋅鋇鋌鋏鋒鋙鋜鋝鋟鋣鋤鋥鋦鋨鋩鋪鋮鋯鋰鋱鋶鋸鋻鋼錁錄錆錇錈錏錐錒錕錘錙錚錛錟錠錡錢錦錨錩錫錮錯錳錶錸錼鏨鍀鍁鍃鍆鍇鍈鍊鍋鍍鍔鍘鍚鍛鍠鍤鍥鍩鍫鍬鍰鍵鍶鍺鍼鍾鎂鎄鎇鎊鎋鎔鎖鎘鎚鎛鎡鎢鎣鎦鎧鎩鎪鎬鎮鎰鎲鎳鎵鎸鎿鏃鏇鏈鏌鏍鏐鏑鏗鏘鏚鏜鏝鏞鏟鏡鏢鏤鏨鏰鏵鏷鏹鏽鐃鐋鐐鐒鐓鐔鐘鐙鐝鐠鐦鐧鐨鐫鐮鐲鐳鐵鐶鐸鐺鐿鑄鑊鑌鑑鑒鑔鑕鑛鑞鑠鑣鑤鑥鑪鑭鑰鑱鑲鑷鑹鑼鑽鑾鑿钁钂釓釔針釘釗釙釕釷釺釧釤鈒釩釣鍆釹鍚釵鈃鈣鈈鈦钜鈍鈔鐘鈉鋇鋼鈑鈐鑰欽鈞鎢鉤鈧鈁鈥鈄鈕鈀鈺錢鉦鉗鈷缽鈳鉕鈽鈸鉞鑽鉬鉭鉀鈿鈾鐵鉑鈴鑠鉛鉚鈰鉉鉈鉍鈮鈹鐸鉶銬銠鉺鋩錏銪鋮鋏鋣鐃銍鐺銅鋁銱銦鎧鍘銖銑鋌銩銛鏵銓鎩鉿銚鉻銘錚銫鉸銥鏟銃鐋銨銀銣鑄鐒鋪鋙錸鋱鏈鏗銷鎖鋰鋥鋤鍋鋯鋨鏽銼鋝鋒鋅鋶鉲鐧銳銻鋃鋟鋦錒錆鍺鍩錯錨錛錡鍀錁錕錩錫錮鑼錘錐錦鑕鍁錈鍃錇錟錠鍵鋸錳錙鍥鍈鍇鏘鍶鍔鍤鍬鍾鍛鎪鍠鍰鎄鍍鎂鏤鎡鐨鋂鏌鎮鎛鎘鑷钂鐫鎳錼鎦鎬鎊鎰鎵鑌鎔鏢鏜鏝鏍鏰鏞鏡鏑鏃鏇鏐鐔钁鐐鏷鑥鐓鑭鐠鑹鏹鐙鑊鐳鐶鐲鐮鐿鑔鑣鑞鑱鑲長長門閂閃閆閈閉開閌閎閏閑閒間閔閘閙閡関閣閤閥閨閩閫閬閭閱閲閶閹閻閼閽閾閿闃闆闈闊闋闌闍闐闒闓闔闕闖闚關闞闠闡闢闤闥門閂閃閆閈閉問闖閏闈閑閎間閔閌悶閘鬧閨聞闥閩閭闓閥閣閡閫鬮閱閬闍閾閹閶鬩閿閽閻閼闡闌闃闠闊闋闔闐闒闕闞闤隊阬阯陽陰陣階際陸隴陳陘陝陘陝陞陣隉隕險陰陳陸陽隂隄隉隊階隨隱隕隖際隣隨險隱隴隸隷隸隻雋難雋雛雖雙雛雜雞讎離難雲靂電霧霽黴霑霛霤霧靄霽靂靄靆靈靉靚靜靚靜靣靨靦靨靭鞀鞉鞏韃鞽鞦韉韝鞽韁韃韆韉韋韌韍韓韙韜韝韞韤韋韌韍韓韙韞韜韻韻響頁頂頃項順頇須頊頌頎頏預頑頒頓頗領頜頡頤頦頫頭頮頰頲頴頷頸頹頻頽顆題額顎顏顒顓顔願顙顛類顢顥顧顫顬顯顰顱顳顴頁頂頃頇項順須頊頑顧頓頎頒頌頏預顱領頗頸頡頰頲頜潁熲頦頤頻頮頹頷頴穎顆題顒顎顓顏額顳顢顛顙顥纇顫顬顰顴風颭颮颯颱颳颶颸颺颻颼飀飃飄飆飇飈風颺颭颮颯颶颸颼颻飀飄飆飆飛飛飢飣飥饗飩飪飫飭飯飲飴飼飽飾飿餃餄餅餉養餌饜餎餏餑餒餓餕餖餘餚餛餜餞餡館餱餳餵餶餷餺餼餽餾餿饁饃饅饈饉饊饋饌饑饒饗饜饝饞饢飣饑飥餳飩餼飪飫飭飯飲餞飾飽飼飿飴餌饒餉餄餎餃餏餅餑餖餓餘餒餕餜餛餡館餷饋餶餿饞饁饃餺餾饈饉饅饊饌饢馬馭馮馱馳馴馹駁駐駑駒駔駕駘駙駛駝駟駡駢駭駮駰駱駸駿騁騂騅騌騍騎騏騐騖騗騙騣騤騫騭騮騰騶騷騸騾驀驁驂驃驄驅驊驌驍驏驕驗驘驚驛驟驢驤驥驦驪驫馬馭馱馴馳驅馹駁驢駔駛駟駙駒騶駐駝駑駕驛駘驍罵駰驕驊駱駭駢驫驪騁驗騂駸駿騏騎騍騅騌驌驂騙騭騤騷騖驁騮騫騸驃騾驄驏驟驥驦驤骾髏髖髕髏髒體髕髖髩髮鬁鬆鬍鬢鬚鬢鬥鬦鬧鬨鬩鬭鬮鬱魘魎魎魘魚魛魢魨魯魴魷魺鮁鮃鮊鮋鮌鮍鮎鮐鮑鮒鮓鮚鮜鮞鮦鮪鮫鮭鮮鮳鮶鮷鮺鯀鯁鯇鯉鯊鯒鯔鯕鯖鯗鯛鯝鯡鯢鯤鯧鯨鯪鯫鯰鯴鯵鯷鯽鯿鰁鰂鰃鰈鰉鰌鰍鰏鰐鰒鰓鰛鰜鰟鰠鰣鰥鰨鰩鰭鰮鰱鰲鰳鰵鰷鰹鰺鰻鰼鰾鱂鱅鱈鱉鱒鱓鱔鱖鱗鱘鱝鱟鱠鱣鱤鱧鱨鱭鱯鱷鱸鱺魚魛魢魷魨魯魴魺鮁鮃鯰鱸鮋鮓鮒鮊鮑鱟鮍鮐鮭鮚鮳鮪鮞鮦鰂鮜鱠鱭鮫鮮鮺鯗鱘鯁鱺鰱鰹鯉鰣鰷鯀鯊鯇鮶鯽鯒鯖鯪鯕鯫鯡鯤鯧鯝鯢鯰鯛鯨鯵鯴鯔鱝鰈鰏鱨鯷鰮鰃鰓鱷鰍鰒鰉鰁鱂鯿鰠鼇鰭鰨鰥鰩鰟鰜鰳鰾鱈鱉鰻鰵鱅鰼鱖鱔鱗鱒鱯鱤鱧鱣鳥鳧鳩鳲鳳鳴鳶鴆鴇鴉鴒鴕鴛鴝鴞鴟鴣鴦鴨鴬鴯鴰鴴鴻鴿鵂鵃鵐鵑鵒鵓鵜鵝鵞鵠鵡鵪鵬鵮鵯鵲鵶鵷鵾鶇鶉鶊鶓鶖鶘鶚鶡鶤鶥鶩鶬鶯鶲鶴鶹鶺鶻鶼鶿鷀鷁鷂鷄鷊鷓鷖鷗鷙鷚鷥鷦鷫鷯鷰鷲鷳鷴鷸鷹鷺鷼鸇鸌鸎鸏鸕鸘鸚鸛鸝鸞鳥鳩雞鳶鳴鳲鷗鴉鶬鴇鴆鴣鶇鸕鴨鴞鴦鴒鴟鴝鴛鴬鴕鷥鷙鴯鴰鵂鴴鵃鴿鸞鴻鵐鵓鸝鵑鵠鵝鵒鷳鵜鵡鵲鶓鵪鶤鵯鵬鵮鶉鶊鵷鷫鶘鶡鶚鶻鶖鶿鶥鶩鷊鷂鶲鶹鶺鷁鶼鶴鷖鸚鷓鷚鷯鷦鷲鷸鷺鸇鷹鸌鸏鸛鸘鹵鹹鹺鹼鹽鹺麅麗麥麥麩麪麯麴麵麩麼麽黃黃黌黌點黶黨黷黲黲黴黶黷黽黽黿鼂鼇鼈鼉黿鼂鼉鼕鞀鼴鼴齇齇齊齋齎齏齊齏齒齔齕齗齙齜齟齠齡齣齦齧齪齬齲齶齷齒齔齕齗齟齡齙齠齜齦齬齪齲齷龍龐龔龕龍龔龕龜龜]+'

# traditional_char_regular = r'[萬]+'

logging.getLogger(__name__)

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(filename)s[%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%H:%M:%S')



class LM_Config(object):
    max_seq_length = CONFIGS["general_configs"]["lm"]["max_seq"]
    vocab_file = CONFIGS["general_configs"]["lm"]["vocab"]
    bert_config_file = CONFIGS["general_configs"]["lm"]["bert_configs"]
    if CONFIGS["general_configs"]["lm"]["fine_tuning_is_on"]:
        init_checkpoint = CONFIGS["general_configs"]["lm"]["fine-tuned"]
    else:
        init_checkpoint = CONFIGS["general_configs"]["lm"]["pre-trained"]
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)
    topn = CONFIGS["general_configs"]["lm"]["top_n"]


class Filter(object):
    def __init__(self):
        self.curve_idx_sound = {0: {0: Curves.curve_null,  # 0 for non-difference
                              1: Curves.curve_null,
                              2: Curves.curve_null,
                              3: Curves.curve_null,
                              4: Curves.curve_null,
                              5: Curves.curve_null,
                              6: Curves.curve_null,
                              7: Curves.curve_null,
                              },
                          1: {0: Curves.curve_null,  # 1 for difference
                              1: Curves.curve_null,
                              2: Curves.curve_null,
                              3: Curves.curve_null,
                              4: Curves.curve_null,
                              5: Curves.curve_null,
                              6: Curves.curve_null,
                              7: Curves.curve_null,
                              }}

        self.curve_idx_shape = {0: {0: Curves.curve_null,  # 0 for non-difference
                                    1: Curves.curve_null,
                                    2: Curves.curve_null,
                                    3: Curves.curve_null,
                                    4: Curves.curve_null,
                                    5: Curves.curve_null,
                                    6: Curves.curve_null,
                                    7: Curves.curve_null,
                                    },
                                1: {0: Curves.curve_null,  # 1 for difference
                                    1: Curves.curve_null,
                                    2: Curves.curve_null,
                                    3: Curves.curve_null,
                                    4: Curves.curve_null,
                                    5: Curves.curve_null,
                                    6: Curves.curve_null,
                                    7: Curves.curve_null,
                                    }}

    def filter(self, rank, difference, error, filter_is_on=True, sim_type='shape'):
        if filter_is_on:
            if sim_type == 'sound':
                curve = self.curve_idx_sound[int(difference)][rank]
            else:
                # print(int(difference))
                curve = Curves.curve_02  # 懒得改上面的字典，先写死，忽略difference
                # curve = self.curve_idx_shape[int(difference)][rank]
        else:
            curve = Curves.curve_null    # curve = Curves.curve_null return true for any case

        if curve(error["confidence"], error["similarity"]) and self.special_filters(error):
            return True

        return False

    @staticmethod
    def special_filters(error):
        """
        Special filters for, essentially, grammatical errors. The following is some examples.
        """
        # if error["original"] in {'他': 0, '她': 0, '你': 0, '妳': 0}:
        #     if error["confidence"] < 0.95:
        #         return False
        #
        # if error["original"] in {'的': 0, '得': 0, '地': 0}:
        #     if error["confidence"] < 0.6:
        #         return False
        #
        # if error["original"] in {'在': 0, '再': 0}:
        #     if error["confidence"] < 0.6:
        #         return False

        return True


class Curves(object):
    def __init__(self):
        pass

    @staticmethod
    def curve_null(confidence, similarity):
        """This curve is used when no filter is applied"""
        return True

    @staticmethod
    def curve_full(confidence, similarity):
        """This curve is used to filter out everything"""
        return False

    @staticmethod
    def curve_01(confidence, similarity):
        """
        we provide an example of how to write a curve. Empirically, curves are all convex upwards.
        Thus we can approximate the filtering effect of a curve using its tangent lines.
        """
        flag1 = 20 / 3 * confidence + similarity - 21.2 / 3 > 0
        flag2 = 0.1 * confidence + similarity - 0.6 > 0

        if flag1 or flag2:
            return True

        return False

    @staticmethod
    def curve_02(confidence, similarity):  # this one is mine
        # similarity > 0.6 or confidence ~ 0.99
        # flag1 = 20 / 3 * confidence + similarity - 21.2 / 3 > 0
        # flag2 = 0.1 * confidence + similarity - 0.67 > 0
        flag1 = 0.3 * confidence + similarity - 0.85 > 0
        flag2 = 0.8 * confidence + similarity - 1.2 > 0
        logging.debug('满足第一个条件：%s', flag1)
        logging.debug('满足第二个条件：%s', flag2)

        if flag1 or flag2:
            return True

        return False


class SpellChecker(object):
    def __init__(self):
        self.masked_lm = MaskedLM(LM_Config())
        self.filter = Filter()

    @staticmethod
    def pass_ad_hoc_filter(corrected_to, original):
        if corrected_to == '[UNK]':
            return False
        if '#' in corrected_to:
            return False
        if len(corrected_to) != len(original):
            return False
        if re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', corrected_to):
            return False
        if re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', original):
            return False
        # 增加：如果correct_to是繁体字，则不接受纠错，return False
        if re.findall(traditional_char_regular, corrected_to):
            return False
        return True

    def get_error(self, sentence, j, cand_tokens, rank=0, difference=True, filter_is_on=True, weights=WEIGHTS, sim_type='shape'):
        """
        对bert得到的每个candidate进行处理，判断是否是错误
        PARAMS
        ------------------------------------------------
        sentence: sentence to be checked
        j: position of the character to be checked  e.g. 6
        cand_tokens: all candidates   e.g. [('主', 0.9994), ('广', 0.0002)]
        rank: the rank of the candidate in question   # in question: being discussed
        filters_on: only used in ablation experiment to remove CSD
        weights: weights for different types of similarity
        sim_type: type of similarity

        """

        cand_token, cand_token_prob = cand_tokens[rank] # cand_token = '主', cand_token_prob = 0.9994

        if cand_token != sentence[j]: # '主' != '丰'
            error = {"error_position": j,
                     "original": sentence[j],
                     "corrected_to": cand_token,
                     "candidates": dict(cand_tokens),
                     "confidence": cand_token_prob,
                     "similarity": CHAR.similarity(sentence[j], cand_token, weights=weights),
                     "sentence_len": len(sentence)}

            if not self.pass_ad_hoc_filter(error["corrected_to"], error["original"]):
                logging.info(f'{error["original"]}'
                             f' --> <PASS-{error["corrected_to"]}>'
                             f' (con={error["confidence"]}, sim={error["similarity"]}, on_top_difference={difference})')
                logging.info('has bad char')
                return None  # 有特殊字符：空

            else:
                if self.filter.filter(rank, difference, error, filter_is_on, sim_type=sim_type):  # 把检查出来的错再过滤一下
                    logging.info(f'{error["original"]}'
                                 f'--> {error["corrected_to"]}'
                                 f' (con={error["confidence"]}, sim={error["similarity"]}, on_top_difference={difference})')
                    logging.info('accept the correction')
                    return error  # 通过了过滤：返回错误信息

                logging.info(f'{error["original"]}'
                             f' --> <PASS-{error["corrected_to"]}>'
                             f' (con={error["confidence"]}, sim={error["similarity"]}, on_top_difference={difference})')
                logging.info('refuse the correction')
                return None  # 没通过过滤：空

        logging.info(f'{sentence[j]}'
                     f' --> <PASS-{sentence[j]}>'
                     f' (con={cand_token_prob}, sim=null, on_top_difference={difference})')
        return None  # 和原字一样：空

    @staticmethod
    def correction_history(sentence, real_error, candidates):
        '''输出纠错过程，为了优化方便
        real_error: 已知的错误位置及正确字符 {2: '阴', 6: '货'}
        candidates: [[('国', 0.99), ('國', 0.01)],[('际', 0.99), ('家', 0.01)]]
        '''
        history = []
        for p, e in real_error.items():
            cand_analysis = {c[0]:[c[1], CHAR.similarity(sentence[p], c[0])] for c in candidates[p+1]}  # index是p+1
            h = {"error_position": p,
                     "original": sentence[p],
                     "correct": e,
                     "candidates": cand_analysis
                     }
            history.append(h)

        return history

    def make_corrections(self,
                         sentences,
                         real_errors=None,
                         tackle_n_gram_bias=CONFIGS["exp_configs"]["tackle_n_gram_bias"],
                         rank_in_question=CONFIGS["general_configs"]["rank"],
                         dump_candidates=CONFIGS["exp_configs"]["dump_candidates"],
                         read_from_dump=CONFIGS["exp_configs"]["read_from_dump"],
                         is_train=False,
                         train_on_difference=True,
                         filter_is_on=CONFIGS["exp_configs"]["filter_is_on"],
                         sim_union=CONFIGS["exp_configs"]["union_of_sims"]
                         ):
        """
        PARAMS:
        ------------------------------
        sentences: sentences with potential errors
        tackle_n_gram_bias: whether the hack to tackle n gram bias is used
        rank_in_question: rank of the group of candidates in question
        dump_candidates: whether save candidates to a specific path
        read_from_dump: read candidates from dump
        is_train: if the script is in the training mode
        train_on_difference: choose the between two sub groups
        filter_is_on: used in ablation experiments to decide whether to remove CSD
        sim_union: whether to take the union of the filtering results given by using two types of similarities

        RETURN:
        ------------------------------
        correction results of all sentences
        """

        processed_sentences = self.process_sentences(sentences)
        generation_time = 0
        if read_from_dump:
            assert dump_candidates
            topn_candidates = pickle.load(open(dump_candidates, 'rb'))
        else:
            start_generation = time.time()
            topn_candidates = self.masked_lm.find_topn_candidates(processed_sentences,
                                                                  batch_size=CONFIGS["general_configs"]["lm"][
                                                                      "batch_size"])
            end_generation = time.time()
            generation_time += end_generation - start_generation
            if dump_candidates:
                pickle.dump(topn_candidates, open(dump_candidates, 'wb'))

        # main workflow
        filter_time = 0
        skipped_count = 0
        results = []
        print('making corrections ...')
        if logging.getLogger().getEffectiveLevel() != logging.INFO:  # show progress bar if not in verbose mode
            progess_bar = tqdm.tqdm(enumerate(topn_candidates))
        else:
            progess_bar = enumerate(topn_candidates)

        for i, cand in progess_bar:
            logging.info("*" * 50)
            logging.info(f"spell checking sentence {sentences[i]}")
            logging.debug('candidates are:')
            for no, c in enumerate(cand):
                logging.debug(f'No.{no}: {c}')
            sentence = ''
            res = []

            # can't cope with sentences containing Latin letters yet.
            if re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', sentences[i]):
                skipped_count += 1
                results.append({"original_sentence": sentences[i],
                                "corrected_sentence": sentences[i],
                                "num_errors": 0,
                                "errors": []
                                })
                logging.info("containing Latin letters; pass current sentence.")

            else:
                # when testing on SIGHAN13,14,15, we recommend using `extension()` to solve
                # issues caused by full-width humbers;
                # when testing on OCR data, we recommend using `extended_cand = cand`
                # extended_cand = extension(cand)
                extended_cand = cand  # 此处的 cand 就是 masked_lm 出来的 一句话的 candidate，
                # [[('国', 0.99), ('國', 0.01)],[('际', 0.99), ('家', 0.01)]]

                history = self.correction_history(sentences[i], real_errors[i], extended_cand) if real_errors else None

                for j, cand_tokens in enumerate(extended_cand):  # 对文本中每个汉字进行循环，每个汉字的纠错过程记录为一个 error dict
                    # cand_tokens [('国', 0.99), ('世', 0.01)]
                    if 0 < j < len(extended_cand) - 1:  # skip the head and the end placeholders -- `。`
                        # print(j)
                        # char：原来正确的汉字，国
                        char = sentences[i][j - 1]

                        # detect and correct errors
                        error = None

                        # spell check rank by rank
                        start_filter = time.time()

                        for rank in range(rank_in_question + 1):  # 对bert模型给出的每个candidate进行循环，最大循环到第rank个
                            logging.info(f"spell checking on rank={rank}")

                            if not sim_union:  # 肯定是不union的
                                if WEIGHTS[0] > WEIGHTS[1]:
                                    sim_type = 'shape'
                                else:
                                    sim_type = 'sound'
                                logging.debug("original:%s", sentences[i][j - 1])
                                logging.debug("candidate:%s", cand_tokens[rank][0])
                                # difference: bert预测的第一个字，和原字一不一样
                                error = self.get_error(sentences[i],
                                                       j - 1,
                                                       cand_tokens,
                                                       rank=rank,
                                                       difference=cand_tokens[0][0] != sentences[i][j - 1],
                                                       filter_is_on=filter_is_on, sim_type=sim_type)

                            else:  # 所以这里不用看了

                                logging.info("using shape similarity:")
                                error_shape = self.get_error(sentences[i],
                                                             j - 1,
                                                             cand_tokens,
                                                             rank=rank,
                                                             difference=cand_tokens[0][0] != sentences[i][j - 1],
                                                             filter_is_on=filter_is_on,
                                                             weights=(1, 0, 0), sim_type='shape')
                                logging.info("using sound similarity:")
                                error_sound = self.get_error(sentences[i],
                                                             j - 1,
                                                             cand_tokens,
                                                             rank=rank,
                                                             difference=cand_tokens[0][0] != sentences[i][j - 1],
                                                             filter_is_on=filter_is_on,
                                                             weights=(0, 1, 0), sim_type='sound')
  

                                if error_shape:
                                    error = error_shape
                                    if is_train:
                                        error = None  # to train shape similarity, we do not want any error that has already detected by sound similarity
                                else:
                                    error = error_sound

                            if error:   # 这里不明白，rank_in_question 是干什么用的？只要rank0的收时候发现了错误，就不继续看rank1了
                                if is_train:
                                    if rank != rank_in_question:  # not include candidate that has a predecessor already
                                        # taken as error
                                        error = None
                                        # break
                                    else:
                                        # do not include candidates produced by different candidate generation strategy
                                        if train_on_difference != (cand_tokens[0][0] != sentences[i][j - 1]):
                                            error = None
                                else:
                                    break

                        end_filter = time.time()
                        filter_time += end_filter - start_filter

                        if error:
                            res.append(error)
                            char = error["corrected_to"]
                            sentence += char
                            continue

                        sentence += char

                # a small hack: tackle the n-gram bias problem: when n adjacent characters are erroneous,
                # pick only the one with the greatest confidence.
                error_delete_positions = []
                if tackle_n_gram_bias:
                    error_delete_positions = []
                    for idx, error in enumerate(res):
                        delta = 1
                        n_gram_errors = [error]
                        try:
                            while res[idx + delta]["error_position"] == error["error_position"] + delta:
                                n_gram_errors.append(res[idx + delta])
                                delta += 1
                        except IndexError:
                            pass
                        n_gram_errors.sort(key=lambda e: e["confidence"], reverse=True)
                        error_delete_positions.extend([(e["error_position"], e["original"]) for e in n_gram_errors[1:]])

                    error_delete_positions = dict(error_delete_positions)

                    res = [e for e in res if e["error_position"] not in error_delete_positions]

                    def process(pos, c):
                        if pos not in error_delete_positions:
                            return c
                        else:
                            return error_delete_positions[pos]

                    sentence = ''.join([process(pos, c) for pos, c in enumerate(sentence)])

                # add the result for current sentence
                results.append({"original_sentence": sentences[i],
                                "corrected_sentence": sentence,
                                "num_errors": len(res),
                                "errors": res,
                                "history":history
                                })
                logging.info(f"current sentence {sentences[i]} is corrected to {sentence}")
                logging.info(f" {len(error_delete_positions)} errors are deleted to prevent n-gram bias problem")
                logging.info("*" * 50 + '\n')
        try:
            print(
                f"Elapsed time: {generation_time // 60} min {generation_time % 60} s in generating candidates for {len(sentences)} sentences;\n"
                f"              {filter_time} s in filtering candidates for {len(sentences) - skipped_count} sentences;\n"
                f"Speed: {generation_time / len(sentences) * 1000} ms/sentence in generating and {filter_time / (len(sentences) - skipped_count) * 1000} ms/sentence in filtering ")
        except ZeroDivisionError:
            print(
                f"Elapsed time: {generation_time // 60} min {generation_time % 60} s in generating candidates for {len(sentences)} sentences;\n"
                f"              {filter_time} s in filtering candidates for {len(sentences) - skipped_count} sentences;\n"
                f"Speed: {generation_time / len(sentences) * 1000} ms/sentence in generating and NaN ms/sentence in filtering ")
        return results

    def repeat_make_corrections(self, sentences, real_errors=None, num=3, is_train=False, train_on_difference=True):
        all_results = []
        sentences_to_be_corrected = sentences

        for _ in range(num):
            results = self.make_corrections(sentences_to_be_corrected,
                                            real_errors=real_errors,
                                            is_train=is_train,
                                            train_on_difference=train_on_difference)
            sentences_to_be_corrected = [res["corrected_sentence"] for res in results]
            all_results.append(results)

        correction_history = []
        for i, sentence in enumerate(sentences):
            r = {"original_sentence": sentence, "correction_history": []}
            for item in all_results:
                r["correction_history"].append(item[i]["corrected_sentence"])
            correction_history.append(r)

        return all_results, correction_history

    def correction_service(self, sentences, real_errors=None, is_train=False, train_on_difference=True):
        """correction service for CV. only return result sentences;
        for convenience, do not repeat"""
        results = self.make_corrections(sentences,
                                        real_errors=real_errors,
                                        is_train=is_train,
                                        train_on_difference=train_on_difference)

        return [res["corrected_sentence"] for res in results]

    @staticmethod
    def process_sentences(sentences):
        """Because masked language model is trained on concatenated sentences,
         the start and the end of a sentence in question is very likely to be
         corrected to the period symbol (。) of Chinese. Hence, we add two period
        symbols as placeholders to prevent this from harming FASPell's performance."""
        return ['。' + sent + '。' for sent in sentences]


def extension(candidates):
    """this function is to resolve the bug that when two adjacent full-width numbers/letters are fed to mlm,
       the output will be merged as one output, thus lead to wrong alignments."""
    new_candidates = []
    for j, cand_tokens in enumerate(candidates):
        real_cand_tokens = cand_tokens[0][0]
        if '##' in real_cand_tokens:  # sometimes the result contains '##', so we need to get rid of them first
            real_cand_tokens = real_cand_tokens[2:]

        if len(real_cand_tokens) == 2 and not re.findall(r'[a-zA-ZＡ-Ｚａ-ｚ]+', real_cand_tokens):
            a = []
            b = []
            for cand, score in cand_tokens:
                real_cand = cand
                if '##' in real_cand:
                    real_cand = real_cand[2:]
                a.append((real_cand[0], score))
                b.append((real_cand[-1], score))
            new_candidates.append(a)
            new_candidates.append(b)
            continue
        new_candidates.append(cand_tokens)

    return new_candidates

def compare_text(correct, wrong):
    '''比较正确与错误的两句话，返回错误出现的index及字。暂时只考虑替代错误
    对于脱除和增加这两种错误结果是不准确的'''
    errors = {}
    for i, (c, w) in enumerate(zip(correct, wrong)):
        if c == w: continue
        errors[i] = c
    return errors


def repeat_test(test_path, spell_checker, repeat_num, is_train, train_on_difference=True):
    sentences = []
    real_errors = []
    for line in open(test_path, 'r', encoding='utf-8'):
        logging.debug('line: %s', line)
        logging.debug('%s', line.strip().split('\t'))
        num, wrong, correct = line.strip().split('\t')
        sentences.append(wrong)
        real_error = compare_text(correct, wrong)  # {2: '阴', 6: '货'}
        real_errors.append(real_error)

    all_results, correction_history = spell_checker.repeat_make_corrections(sentences, real_errors, num=repeat_num,
                                                                            is_train=is_train,
                                                                            train_on_difference=train_on_difference)
    if is_train:
        for i, res in enumerate(all_results):
            print(f'performance of round {i}:')
            test_unit(res, test_path,
                      f'difference_{int(train_on_difference)}-rank_{CONFIGS["general_configs"]["rank"]}-results_{i}')
    else:
        for i, res in enumerate(all_results):
            print(f'performance of round {i}:')
            test_unit(res, test_path, f'FASPell/test/test-results_{i}')

    w = open(f'history.json', 'w', encoding='utf-8')
    w.write(json.dumps(correction_history, ensure_ascii=False, indent=4, sort_keys=False))
    w.close()

    # 重复纠错，非测试
def repeat_non_test(sentences, spell_checker, repeat_num):
    all_results, correction_history = spell_checker.repeat_make_corrections(sentences, num=repeat_num,
                                                                            is_train=False,
                                                                            train_on_difference=True)

    w = open(f'history.json', 'w', encoding='utf-8')
    w.write(json.dumps(correction_history, ensure_ascii=False, indent=4, sort_keys=False))
    w.close()
    for i, res in enumerate(all_results):
        w = open(f'results_{i}.json', 'w', encoding='utf-8')
        w.write(json.dumps(res, ensure_ascii=False, indent=4, sort_keys=False))
        w.close()


def test_unit(res, test_path, out_name, strict=True):
    out = open(f'{out_name}.txt', 'w', encoding='utf-8')

    corrected_char = 0
    wrong_char = 0
    corrected_sent = 0
    wrong_sent = 0
    true_corrected_char = 0
    true_corrected_sent = 0
    true_detected_char = 0
    true_detected_sent = 0
    accurate_detected_sent = 0
    accurate_corrected_sent = 0
    all_sent = 0

    for idx, line in enumerate(open(test_path, 'r', encoding='utf-8')):
        all_sent += 1
        falsely_corrected_char_in_sentence = 0
        falsely_detected_char_in_sentence = 0
        true_corrected_char_in_sentence = 0

        num, wrong, correct = line.strip().split('\t')
        predict = res[idx]["corrected_sentence"]
        
        wrong_num = 0
        corrected_num = 0
        original_wrong_num = 0
        true_detected_char_in_sentence = 0

        for c, w, p in zip(correct, wrong, predict):
            if c != p:
                wrong_num += 1
            if w != p:
                corrected_num += 1
                if c == p:
                    true_corrected_char += 1
                if w != c:
                    true_detected_char += 1
                    true_detected_char_in_sentence += 1
            if c != w:
                original_wrong_num += 1

        out.write('\t'.join([str(original_wrong_num), wrong, correct, predict, str(wrong_num)]) + '\n')
        corrected_char += corrected_num
        wrong_char += original_wrong_num
        if original_wrong_num != 0:
            wrong_sent += 1
        if corrected_num != 0 and wrong_num == 0:
            true_corrected_sent += 1

        if corrected_num != 0:
            corrected_sent += 1

        if strict:
            true_detected_flag = (true_detected_char_in_sentence == original_wrong_num and original_wrong_num != 0 and corrected_num == true_detected_char_in_sentence)
        else:
            true_detected_flag = (corrected_num != 0 and original_wrong_num != 0)
        # if corrected_num != 0 and original_wrong_num != 0:
        if true_detected_flag:
            true_detected_sent += 1
        if correct == predict:
            accurate_corrected_sent += 1
        if correct == predict or true_detected_flag:
            accurate_detected_sent += 1

    print("corretion:")
    print(f'char_p={true_corrected_char}/{corrected_char}')
    print(f'char_r={true_corrected_char}/{wrong_char}')
    print(f'sent_p={true_corrected_sent}/{corrected_sent}')
    print(f'sent_r={true_corrected_sent}/{wrong_sent}')
    print(f'sent_a={accurate_corrected_sent}/{all_sent}')
    print("detection:")
    print(f'char_p={true_detected_char}/{corrected_char}')
    print(f'char_r={true_detected_char}/{wrong_char}')
    print(f'sent_p={true_detected_sent}/{corrected_sent}')
    print(f'sent_r={true_detected_sent}/{wrong_sent}')
    print(f'sent_a={accurate_detected_sent}/{all_sent}')

    w = open(f'{out_name}.json', 'w', encoding='utf-8')
    w.write(json.dumps(res, ensure_ascii=False, indent=4, sort_keys=False))
    w.close()


def parse_args():
    usage = '\n1. You can spell check several sentences by:\n' \
            'python faspell.py 扫吗关注么众号 受奇艺全网首播 -m s\n' \
            '\n' \
            '2. You can spell check a file by:\n' \
            'python faspell.py -m f -f /path/to/your/file\n' \
            '\n' \
            '3. If you want to do experiments, use mode e:\n' \
            ' (Note that experiments will be done as configured in faspell_configs.json)\n' \
            'python faspell.py -m e\n' \
            '\n' \
            '4. You can train filters under mode e by:\n' \
            'python faspell.py -m e -t\n' \
            '\n' \
            '5. to train filters on difference under mode e by:\n' \
            'python faspell.py -m e -t -d\n' \
            '\n'
    parser = argparse.ArgumentParser(
        description='A script for FASPell - Fast, Adaptable, Simple, Powerful Chinese Spell Checker', usage=usage)

    parser.add_argument('multiargs', nargs='*', type=str, default=None,
                        help='sentences to be spell checked')
    parser.add_argument('--mode', '-m', type=str, choices=['s', 'f', 'e'], default='s',
                        help='select the mode of using FASPell:\n'
                             ' s for spell checking sentences as args in command line,\n'
                             ' f for spell checking sentences in a file,\n'
                             ' e for doing experiments on FASPell')
    parser.add_argument('--file', '-f', type=str, default=None,
                        help='under mode f, a file to be spell checked should be provided here.')
    parser.add_argument('--difference', '-d', action="store_true", default=False,
                        help='train on difference')
    parser.add_argument('--train', '-t', action="store_true", default=False,
                        help='True=to train FASPell with confidence-similarity graphs, etc.'
                             'False=to use FASPell in production')

    args = parser.parse_args()
    return args


def main():
    spell_checker = SpellChecker()
    args = parse_args()
    if args.mode == 's':  # command line mode
        try:

            assert args.multiargs is not None
            assert not args.train

            logging.basicConfig(level=logging.INFO)
            repeat_non_test(args.multiargs, spell_checker, CONFIGS["general_configs"]["round"])

        except AssertionError:
            print("Sentences to be spell checked cannot be none.")

    elif args.mode == 'f':  # file mode
        try:
            assert args.file is not None
            sentences = []
            for sentence in open(args.file, 'r', encoding='utf-8'):
                sentences.append(sentence.strip())
            repeat_non_test(sentences, spell_checker, CONFIGS["general_configs"]["round"])

        except AssertionError:
            print("Path to a txt file cannot be none.")

    elif args.mode == 'e':  # experiment mode
        
        if args.train:
            repeat_test(CONFIGS["exp_configs"]["training_set"], spell_checker, CONFIGS["general_configs"]["round"],
                    args.train, train_on_difference=args.difference)
            # assert not CONFIGS["exp_configs"]["union_of_sims"]  # union of sims is a strategy used only in testing
            name = f'difference_{int(args.difference)}-rank_{CONFIGS["general_configs"]["rank"]}-results_0'
            plot.plot(f'{name}.json',
                      f'{name}.txt',
                      store_plots=CONFIGS["exp_configs"]["store_plots"],
                      plots_to_latex=CONFIGS["exp_configs"]["store_latex"])
        else:
            repeat_test(CONFIGS["exp_configs"]["testing_set"], spell_checker, CONFIGS["general_configs"]["round"],
                    args.train, train_on_difference=args.difference)


if __name__ == '__main__':
    main()