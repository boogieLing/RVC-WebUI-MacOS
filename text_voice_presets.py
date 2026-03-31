"""ChatTTS source-voice presets and constrained tone resolvers for text-to-audio generation."""

from __future__ import annotations

from dataclasses import dataclass
import re

FEMALE_BASE_SPEAKER_EMB = """蘁淰敍欀暄媴帠兌耳蟕臔垳焫埽洘裮豔吊謖堶譎茰绅婙熼牪穜悇厸桜呄椂贱擕泚澤尌慪砶藺觮挡跴枣聞濞欋敆困真捷薋场縺櫣亄娂蝤柌畕朗洧漷燑楢侾棹刯痷殳琅箠孅粶縰站掅蓧諾娴燆潈伝緿繲嬢杻攩赢赵捓再巠猦竑蕵伷名蠼潡蕩哋纓癶濈撆桙恟剁啩略狶恡璅灟匆喈披恡薂桢澨凝婳晅氬淳笸巭曔艠缜恄囷塰俩劣呃狼匉擳玟儾璊拋竗琶窻塒吼焪亝嶅獩喎街猫耙淚兛淂苇樄助携瓘漌貶矯胘瓧芘诌禞叝看紽襮籌瑵奀罐撚削拽蜴殄嵷貼脊繑淣太榗榰湘疘浏榮歀粻帶話瑶堭羇涐滀谖嚋堊毨牂吹旍嬙烇捚杺熿楽蟺婠獠艷芋啟癇巆垪胎突皋濙峉秶蔁磥稉橑侰啷壶油讋孌盡穀腧疪撙爿櫳偅惆熬蔪篸榸瞧戮谪癣氁眕埛訏惲珴扢缫责狔疌憼蓣氻儍扠崘堋趆绗歭硆狮恂坣跤偮巊唡賠楇臟妝贸夂碴病啷曤壦篶聅瓶虠娹藅蚔峤尚蒴赔胶諲楑棝憳祏祲湹撑舊沽渉掑桼狁秜暿乒腶化坵诣扩爞狐砳欹琌剔璺贇謉皌噝汪濯懭硸敶妔方罐傠訫蝩胿懡橝耎瞉玶残毓襀嶅奈怬繌揩碈蘸瓄撻襮藻嘲缨苫舮痟毧絝蕗澰省剣欔蜫蠀礦岪詙菔呸槿盰潊澾薿覴蛧羕枲纣勛歮擜坺攺莳嗨秂複菲品穛榍芲藨啖水俀揋烰誛乁絃穴萜仿焇娬孷搗琔甠嬑昅祍溡夒氌肕翔癨澔嘸暞丟峗菧评勛跷組縐覤諙昦狊壪烮盹蕨圠拎肼燆槭暠荾襟蟔棻褸瞉寙戰幔緵僇冓誷裲莡謭欠谠窏膛棖嶐域稨猑咇舘殳咛箼薎芐拧粸粐晘卞蕸仮扅敕獏厶倹籽夞洤縉济犴玶掰嘏叮兽瑒奓檯襑狏墐葯煰娄蜋抪琽殤茮耻兩臙播嵙僖謫擔爱哪褊趽搦熲薱莍灮槣尴桫揧仅坪罷痪谇榒炨臊倫谮搇変蝌砃缽簪竅彶搻亇姄苠绿急脑粂嶘楝蜝湴欤宒蛐圓悑唓纡蝙桨兾芡涛賘詞嚏豣伈茻扰撅兿蔴籢祬尽罞彜梩砝匶攀蟣貍柢咠弑朩攸筠蝉囫桘劍蓤捽晟淌芅葨粁蔻蛫汉嬤贐奶瑅希賗舴楔併梭谰燓氕彴擂焕聞穓媾區吣炞俩埞絊潇店幆蜲禀蓎膨岀棕曶袷螣嚷斅殖濗註僔啧豇桜痨牢礙詥竞棕媞跳汶萁堋旎厇洲孠晖濹忻扩投晪糶亸懄燜綗翹譊熡傚幧綁巊卄瘏痩乯稀則緪捧挡妒茌湤晎皾嶙謽山揩瘎諽矧琡娅慅瑣缄渒聘塱噟恗貳杏徣埵胳奄牎淤妒訾榓襭彔皣牡痋蔸裏炻揭悫幱滛筂聈党賢嵢层偣埧蛯燬礅劳処巂莝獹攚褵芸胠傱亳板葞燱绁級壭腛汜憝立芺羿萻獌蘚奯翟栖禣畚荳沕唜籿祲烹膆芬奸氞氟潣簳嘎貑简牑葉缡怸礕詇僤藟猉娃趐兗敍偂洋譀帇咑一㴁"""

TEXT_VOICE_PRESETS = {
    "female_core": {
        "prompt": "[speed_4]",
        "refine_prompt": "[oral_2][laugh_0][break_3]",
        "temperature": 0.24,
        "top_P": 0.64,
        "top_K": 16,
        "spk_emb": FEMALE_BASE_SPEAKER_EMB,
    },
    "bright": {
        "prompt": "[speed_5]",
        "refine_prompt": "[oral_2][laugh_0][break_4]",
        "temperature": 0.34,
        "top_P": 0.72,
        "top_K": 24,
        "spk_emb": "蘁淰敭欀富弰絒謢曙榱縲试詄蠆莓喀哣簏洊屻株敥腼獳湣簡攧偟掄刹蝌柞荸活咤袽祬笳獆藆仚剐楍螥収怎卧伱噡繥揝勻膼灯缍標怢惲圣碚种礯挖翖攓襀豥埭褜諽嶒臁嬣喾詋蟅拰褊滒竁橓衟囻榌录氬姀簝猾樔燗婵脧矴摢哃贜倅嗏帶絊倭淝菘砀睹極瀫刹庨綺秳穹玽濆蔤惹褆肋岜烣痌萭抷涟侐捣癄劎籡肆汆藜礰綞忦哣拫蘧筻誃焁蓹獞睪萂嗡坓蒀塌泙桝淼蟼憶娉貴簕皐游欁歧蝴擿攧艋珴範萇箜蒻誌蟀业賕芁兿葦眚筢劚测劽佷瞣薟糉緺诲暑極坿亡愗噯絩斁剦扂焓刚嘟癋婧梊譃惽偫嘗櫰譏瓨梑尤桁磌娺偣曂珞莻詬覾瘲肖芿硋儨乏岈恎腅構耰记嶪祇筿簥掹苙禁費艺僒嗢羗总勹欸仑棌祻議漫構樿真廡朻劎戃蛜竷呇譏摰抃復矎孑婹弭嘢埗篓崚佡甙壸偢耮痰贏篼璎扥渫筼俼爑礧樍萷矚梬猚乪报择词婿絬藈优篿蛊嵻咇催匁抻听譱咽暊摄蠨垅憳椊搜湐傸譗獵裹薫端汑汋坘侪塝勪眊坿弞戭膬簦荙肚燎憪觘賜襍爅擐別劁硬茦枫赏糲烗耓咠許慄谈眯桁孟縖薴蜦柸疐待犃虧潶紇裛簋懟薇啺蜱肆壁绔岷爽偪裧奕携撼薠惵梤宔肦矌揄美浇賈礣揌瘈暉蝢硼袎萿恶擲拫褝彩椨艴櫞烬脨恄撞衴濖嫃湡汳灕曘炢剾孩儁疈廱粲蟤墊犚櫄穰矞暠藞謙蟊他砖屜羜娛棋簑戯诬獔憆搽誱瀹曰碸惒唙燍氓瑇珝緇义嵠珛垶枒葫樨孲测藶妌毓茢僤慅狸泧柣豇嶹匿蝬伣悋柬謓芄誥乖訕嵞喞岍爙嶍緈褖乌蔍剕瞒礫胲焫猙愈憶喧唃塵褩港繌訕汬啷稴困問抯臓抭诀粀獎擎篔襄嚫寘杘荝屈蟦砐怋殌偒姷枠烾壍謞婯俜囊柔挆稒登豂垊袺咹憒橣茪瞬圽暗蘼希檆昿儯具嶕詰般烠糙誢枈翼嚓咴珦缧見嗐諈櫩蛯侠荛腔吪摊趸橙暈茵埬纖焊禟剻珘犗澥蛭挋父凞孜劅峭函珎葉痔奕蛂橭官淯倍疦杶棅嚮歉吽诜圚桗桬谐缚嫌揼屸棛寜懚珛剭淠傕蔃晁嗱沜源垻荂纁憪噄彖乳紴膜簻湡詟杓卜喁碶吚帆藁址勍漢爎沝犥惃沇芬廏亢呦绾瘁糈绲俄幜徸瞇涑臒粂蜷媽螏瞓淆覃厧枷楲夃潀琑眳崓诌籃芠蔊膯崒聵军胓槽箈葚付璝噮詫泄蕳檶槧竸峄灧崠浨艌脼堐蛌被碃瓴婖无舠煰礦腂撱蘎窖圾搵官惧悇赨碴裫嬲潗僕嚇僌沌賰两炌圐肉粀樾灀岩凃榁忟臁抚伅訥爳谵嘺蘜熜槇桐籴瓩皋赕惃暛玁烅艜亍又伀伯氖哳桦柸亼战嶱漴蔑嵢凕羝腨庪舏灊光笢氬坪汓渧褰榺庙咲奨样探啗冚翁绵昛叔甧舜煃箢蜹缵藕荪莁蕐忹散盻巭蘭墔箖憧袛誌挟苀圯玚亰楀一㴂",
    },
    "airy": {
        "prompt": "[speed_5]",
        "refine_prompt": "[oral_1][laugh_0][break_5]",
        "temperature": 0.28,
        "top_P": 0.68,
        "top_K": 20,
        "spk_emb": "蘁淰敡欀欋晰徎神爣抬攔糨秋藐喑甌溧御苒襾哄曍炫蠞懀篦岹圾欲囱渄丹滋佁湚犗玝柦蔐茤苄悔赩憈卽姠劌引将写虀蓈蕅芲繶伞栙估湬瓭蠀梹羦忪欨巋乌孾暄歆貁俆脏惡繩眗薄笚怗壀漨翂悯惍桶谧藺蘨扤裨札焢纏讒彳訁粞竈槶亱絒夞棪巓搓禸呆尯卸揋潩禐岌焝棚啁性蔆甉俴笺懶詧泷灗瘐舺藕泡源塸硦西千愯菙烷弓惻讆柨摵觽伒梱彧煺礲峛帺稵糳螹萖絇啋咸愙贪犟荥褗笂袙倧燢緃淼虜畇玛浍松覂胛训綼拰稊稡穌曗笻課羫匳争煯橐蠉谦兠潤崍渕琵睝浪耮嬎樎愉贆蝷杷汼滟砞曬宨刳矅熇紃裫脑詨瀲疧檺籆們禩彑磚丠父剠缄秮澐旘翨蒄罊亢椷办覲憽勊脥谨瀞荪剔灏怗堄宎冰煹帕坭倯母燨嘐葯湁浄糽谄灤南跮袙臏謣缗譹洤堪絎喻弤螧缩喛嬦笭壳信扸謞螒徴臵昋毹寣襦晧忻荑惊擱位桰槜祏唦蓲桜蕦藏喠柵濜劺詹浔斴渄唧葓穽哚墔居姇謳蛷緡兪罒蜷签撦娔毆疚緦熄欟媚圷垬覴惂烻礯粏竗匘覨洹奂撢羕赋絚豩爙痶熪憉恖愧巊佌蚺叽斦拀舭厃诜榱蚵絋苟簅疆嶥毶詐侬婡豗嶥庂呁茿棏楧玌璤琱九藭詜乜娸廷暧烲傀烪拀楜礅到蓾壊槚灰参粃为濳竬救耭書糋旅葨練欦荃璝翧橪唸烚瓘奯穳欫匧润傼礰謭搑蚲捔嵼噹幏臅榞呭腌嫘犀椢琀艙苣忭裮佀芍偎媌屚蟣嬐爒旬沗奮翊艙碫娝綁暙袒瑰曾勀槈虽耉秕篅俽諼柦噰苃脰想薉瓫歕宓胘玺穌溗巤搒趝弜执堎漑罓沁査卜徊咷兲勃去夆沤岛墛莨芾蘩砣烁佯狷嫐睢燅琰乺媐崻瓆皳櫺斸棂侻湛垴匁諮璘塶硐腜忭賬绢人侂蕈縼箩気譮術虃乗吣勴成硝嬜嚖缇夑艦秨敼暽皮柆坚又硯坩篻仙痔肸牖襜橐猍睝揻苖窇致褼崳婔橋跂粹薤戀蠥謱副囥千畅冄瓾壷繖氄璻裞誛胂衇走囏珟絖催玌狰碨虇繝礒赯宦襴武赩玍畸怽愰线嬲娽婛柤憎砇艨尻幰勦噸璐淕墠匴漷眹禸埙蒠燝岃價縒暭噜坍喂燙褪仠乛掶挬丝幵虱嵾壤裧悸擾睍泿么壚佲蒍癫戳廫暿硷緰褩絗箄寴挪詽甴旡巳潯潘庁杫嵱賅柌毹矿暒栕瀌焆呖疵璹彌珅澻湂暘碮咃焾碙蟞睵尒濺蘡灕蓤豶婯蓑厩螅佬篥渝墎劊蘛宦屣澁忯藙撤嗍渕簼皒作滃便濘偧瘵渶橢尷搘穇流犟諊茞凗蚣拑徱秹芚屡旙凯惔擾劢場訌嗼杯磕溋噄烱滊淞豑濭敤詙嶘奿搐硫蕾窰澩覐廈蚠艎渨复倴催丼扡荃濟屲灘皫歙妦纕蘆檀舔晛夝噧疳瘾歐娞滱栭泣磽愞楮揦弥吂瀧熊幼孒结渪潄剈倡傎泬橦幉籅码成櫖硷嵉姩啙绚岨艌攓瀘仨一㴆",
    },
    "warm": {
        "prompt": "[speed_4]",
        "refine_prompt": "[oral_2][laugh_0][break_3]",
        "temperature": 0.26,
        "top_P": 0.67,
        "top_K": 18,
        "spk_emb": "蘁淰敝欀徃槈彴炩僖碘噆勔甆浐猼濧憬汩繌棋暲奃务某篱諀措瘘欍峐痐米毷皒漩擙譵敋抰税恱茊套櫱泳垠栚攙搌訟耟礏朦禪廛块挴漣凴歍虜岒勀艊絥瞔初滥庌茞卖榲牍焟丑烏縂獻棈煘苙蔖喅垙絽唅瑼蓸贑炊姵與椰熇瑌純谒槺俤給胔萮塶臧棸箄虆则楌爊胏學观硔檄洉瑤戵啩撊疿诞筞圈獄呁寁筓胅嫒盃垻熣虞猹琦號井跂嵜燭亝泧憘漏讙呦冓笚怚撀巃乛規卿番讀耪畖覙期燓葨嗚櫕茯们胈斣湗翜嬆絫嫢磤冬季枼衢忛胁詚塤簟橣妅潀皥寄匨咘誯匔瘴牬淅丶墿兇窭塐搥笫赮脘恇惵嚍刡膔禱簹腥悂坠欬嗉硟勤湛糣慢薐棘虀斚趂嗑嗅拺緧蝿豫諛疥喷耰弒屵佹謜羅榲弣聂戽歲趪羕埣奀衒奸衁桸堭蚗笎籱令籿樣公撔槆熓噅岕吿蕚呣嶾噺衖挢瑞剅蕷仵櫽嚔菕抌粶伃晿硑嘿咽薬卿捷孝滧営峆绳衴炊賐儝犪纹涭趶臸簟嵻倚敞撐豇咝罙詁愪愳啤胍穞怓紞芬性磉穥核瞡暜荽纛蓇碻捴蜍衶暌杆莀痔磞纩揻慾蚻婉哬嬥矼恬唭戢嘹獍胳剤煙塾葞槑諶萞挑痣籍笖曘捑朲疷赃泊槤曪腦尳劒娈竦傕觨揤艳宀嗊左琭起缿厳淕帤媰塥斷蔗葾焺眼蓆艪萊苍凰潟賏兪涊擥簮互偧粲籔匀璝嘾噷焢樣嬫摭杸敶笈贯碦忢觿應焩磔獃曓丌栎暵孳椪觭譯満祓碢瓵它嫁欍尌衟艛徻檧挷蚔侰斘萮擿窺咁瓵蘍虆奫薧枷俲椿詸瓧胢萳衘洇奢帷瘺共汴吸勅怅紗勉檂悥拰怖谭晆幙蛇蜊漣纣笾盓脉峏佢嚚厼唫姈繤腖挑崠礻崦媱瓢睌櫜冋惚攮坍眎挔澏舂擏搹杈愔籷燓牄析碐牅懠俍字眩嗌稻埅唶羯咗汀态篣睰獈牚貛亙狐恔肻牳瘒穚攪若籲忧祵簳嫿椼偑个榾滋系祕箿蝥砽衺蚰琓蕅桿峂敳搜失绑椁獪臲禑湦簤熐桟幎癦岿茿弻粠僼緈屷絫袧榞燓囀母讷尫蜡宰嗐儈坈掘號臒巬俐焔洕犛憃尤幽艱瑸谇谇浌旵悻栙簖倹虗擌恵睅旒么昀薈薴嗥窋圫嗴柯拠蚖扰曣棂梾聊撇旼耺筘儭礦晈詹勧腪歹熈椓熐繳燎礊質艰搂喷紮懢绡翾穼殣傩燒葘侩佅嗯偆杙獾媾糾萵牅儒柉倍袔硢沃栁淂偂眡睯卻寞捫繑瀳勇匏蓎詠俑喔偊讚試丧唊箒栽肨勋讬厱榴舓忢荾謐痈墎猨焜蝾婗簼茒確讷冔澮漦敒蘦谀瘼蟏范淒侧萛瑑篮俬嚁蒆瓧壂盡泎跥琙畟橓宯賣硠彇灀嚐蒬弴毦莬蘣粒呃槲旘刞累叫氝艙縣昽擹煒浘蜯娣媐童毻菭団漕礄瓻濊舶掬熪皳畼厣庿誗南筩蒿箪斱嬦綁襺締谉喓抾疦柏説穠嵓漓杧紋犔砩特浚炅筅糆燽拢咲丽掎埑芭剿亁縺浽裬睺搅墚蛦眻潃帀㴅",
    },
    "deep": {
        "prompt": "[speed_4]",
        "refine_prompt": "[oral_1][laugh_0][break_2]",
        "temperature": 0.22,
        "top_P": 0.64,
        "top_K": 16,
        "spk_emb": "蘁淰敵欀侌噧憧堷橽楅亐珷忈莒缤品夰寷布刯俦斻僿赘倽荟瑺熴昒墆稲儧沢薹夛河礫叻敨乤蜥翩磞燚咓腻礕攡衂撄垱剬欢寍纾晜厕漰艷班斿声秦胯敭拶檮嚉蜺臺从皓撼諬浻冣伜嘿脺扃瑼胧盖綸憬瑅坝弖癊羈徟蜔盀涢峊排疳祥聶舉恾綕崳筼癡趈嬁眩簊昵扃跆瞨經袽吘庝豑岱暶橡瀽疮俬匞兾俍敊蜙譒旄磛人玙紿蜥勸扥喑埵玲哂藡水棹跧纷螚撴絲蚓抏晉寍舱茞稱歈冘漴爔該滚叓搫炠牕舚乘欂擿欲娵兂岍惞惍嘠誽嵞葹妎呗舻媐训蘢后畧攟秤莄向崲毛盅巭宴撳羪桳賹卂螈恶薯脷澦箮灮俛呅剋揅攰荕粒廙斳茕會謟紾茍傃塀繆异瞓煣恺剬怈贓櫲圸曷瓻洡綷湯菳肏堐倢枽司妋蠽婃灭拕济檼性襥等劙兿灐芺偰爞换演矖倦缕椌擃擳埽戁拋味檾貺荠卓湔粮憔稬衑虊贿玢却襭毬祾怕炭蜦瀍泾揷癛榐互笄礸狰塒傜勊灌璧墫袴跩夫謄澒爁螾紐焍悕攋訜豢砓爳嘓恓煓戢婔緡屇縋凄謎蜳訥嗱湆諬壉炳惩犻蓉貔糙蝝盥蜸蝵瀋噔撟玔丨娒劄婷忒焜瑪缾単華媄卹滥漬喩娑蕈浻覣摴壍摷墥媎嗃罉嶭衽彃埁吓杂臐嵅渆胤帺莤沽瘀荕晗橙膺紡掕圊休憟伱挫咕繸爦腯欳叟薔洅璹峉叱毐拷嵯丷攅漭熕漦撯脑朴瓯卸虔澂兤咂蓥敊恴猒疷巌渔胱湧亜紨絅蝜瘍祄盈呎褍敫灀瞉揩冟杲傺贮澪嫕篒缃赤埲症悔嫚扶稏栿獜漨箦刈菿解菺嵞玣豗膳濽氘肭殯戥所觪眣贠獷赈涒渓泓兖绰檀僳滩冺抏欖憜緖瀳便婺梢豠劙菺嚆嚔硠礿节怃圲碓懃瓴床襰襵杙則浍渓蛋换甙謍蘒贾峮椳訔曥睧涉嵿僣廨螎斲糢朢詎豒侹眳捯塗侺囉偈厌桦忻梼坺瑊粝博桨殐楿狏珹篱殹蒁嫋歓禬罎譵棇敦橾聉吨趡秎堪蝮摍墜撎廚溟皆綅兠啴菦揕爝臻挝神襓臧葚剚柪猏哤峑瞩兰橏冞肔曯团壂偠蕒氽垘縚罧皑兑捯眿脕沙菍虴坬舱蛔呇融硪榿嬴婢秔嚰怦祲盬纨嘤毝渳叹獻瓴榻敦廐滵詠毵珨榚狐紦膽嶒瓣褟灬徂臹梕宯塘吞媶蟝炥礏瞉住膩庞嬗惩罕荠坭蒔蕣煚襣懒殬緯壴箖渄勒冿宛佥徇聲捖瑶砂產潨煠矙腘帨巬浉悷毗乤暥觸氬繎漻蒥嘃耮犨氬庄葋帗佯蒷泼剱傋茎巩缚窜厈熴扊旳潼艳虙贋蘃穗覼訔甛穈礙堭晚濽聮恗檕襋惷塙囥敫蚊誆盖綳寪匸庲蔷竩企疡礸怋窦論螫芈籸屄蒐盓煹圓棕孳懃蘆蟞癸儑膔便熣拪沖葊廻梈橓胸昌烚找匞沥袼潄嚦擇藝爨褱瘜拻蜟孊贰峴侕惥爏囱苘灞趵毧蓸婛羭拫聍焺绞沂程澅暑樓沟喿班淠葾衢橡椸砅圯澁榤菉旓棯尡枈渀一㴄",
    },
}


@dataclass(frozen=True)
class TextVoiceProfile:
    """描述一次文本转语音要使用的具体 ChatTTS 配置。"""

    prompt: str
    refine_prompt: str
    temperature: float
    top_p: float
    top_k: int
    spk_emb: str
    resolved_preset_id: str
    resolved_tone_label: str


@dataclass(frozen=True)
class TextTonePresetSpec:
    """把新 UI tone preset 映射到现有 ChatTTS 基础 speaker 与 prompt。"""

    gender: str
    base_voice_preset: str
    label: str
    prompt: str
    refine_prompt: str
    temperature: float
    top_p: float
    top_k: int


TEXT_TONE_PRESETS: dict[str, TextTonePresetSpec] = {
    # Female presets now share one measured higher-F0 baseline so source speech remains recognizably female,
    # while downstream RVC still owns the final timbre identity and the preset tokens focus on prosody only.
    "female_sad_youth": TextTonePresetSpec("female", "female_core", "Sad Youth", "[oral_2][laugh_0][break_4][speed_3]", "[oral_1][laugh_0][break_4]", 0.22, 0.62, 14),
    "female_gentle": TextTonePresetSpec("female", "female_core", "Gentle", "[oral_2][laugh_0][break_5][speed_4]", "[oral_1][laugh_0][break_4]", 0.24, 0.63, 14),
    "female_elegant_mature": TextTonePresetSpec("female", "female_core", "Elegant Mature", "[oral_2][laugh_0][break_2][speed_4]", "[oral_2][laugh_0][break_3]", 0.23, 0.64, 15),
    "female_bright_cheerful": TextTonePresetSpec("female", "female_core", "Bright Cheerful", "[oral_8][laugh_2][break_1][speed_8]", "[oral_2][laugh_1][break_4]", 0.36, 0.80, 28),
    "female_natural": TextTonePresetSpec("female", "female_core", "Natural", "[oral_2][laugh_0][break_2][speed_4]", "[oral_1][laugh_0][break_3]", 0.20, 0.58, 10),
    "female_heartbroken": TextTonePresetSpec("female", "female_core", "Heartbroken", "[oral_1][laugh_0][break_7][speed_2]", "[oral_1][laugh_0][break_6]", 0.18, 0.54, 10),
    "male_warm_solid": TextTonePresetSpec("male", "warm", "Warm Solid", "[speed_4]", "[oral_2][laugh_0][break_3]", 0.24, 0.66, 18),
    "male_clear_lead": TextTonePresetSpec("male", "warm", "Clear Lead", "[speed_5]", "[oral_1][laugh_0][break_3]", 0.25, 0.67, 18),
    "male_deep_anchor": TextTonePresetSpec("male", "deep", "Deep Anchor", "[speed_4]", "[oral_1][laugh_0][break_2]", 0.22, 0.64, 16),
    "male_calm_narration": TextTonePresetSpec("male", "deep", "Calm Narration", "[speed_4]", "[oral_1][laugh_0][break_3]", 0.23, 0.65, 16),
}


def default_tone_preset_id_for_gender(gender: str | None) -> str:
    """在缺失显式 tone preset 时，根据性别返回稳定的基线 preset。"""

    if (gender or "").strip().lower() == "male":
        return "male_warm_solid"
    return "female_natural"


def _resolve_tone_preset_spec(gender: str | None, tone_preset_id: str | None) -> TextTonePresetSpec:
    """把传入的 tone preset 校验到当前性别允许的集合里。"""

    preset_id = (tone_preset_id or "").strip().lower()
    if preset_id in TEXT_TONE_PRESETS:
        spec = TEXT_TONE_PRESETS[preset_id]
        if spec.gender == (gender or spec.gender).strip().lower():
            return spec
    return TEXT_TONE_PRESETS[default_tone_preset_id_for_gender(gender)]


def _resolve_custom_tone_rule(gender: str, custom_tone_text: str) -> tuple[str, str, float, float, int, str]:
    """把自由文本语气收束成受控规则，避免自定义语气冲掉目标音色匹配。"""

    normalized_text = custom_tone_text.strip().lower()
    if any(keyword in normalized_text for keyword in ("narration", "narrator", "story", "旁白", "讲述")):
        if gender == "male":
            return "[speed_4]", "[oral_1][laugh_0][break_3]", 0.23, 0.65, 16, "Narration"
        return "[speed_4]", "[oral_2][laugh_0][break_3]", 0.26, 0.67, 18, "Narration"
    if any(keyword in normalized_text for keyword in ("deep", "low", "serious", "steady", "沉", "稳", "严肃")):
        if gender == "male":
            return "[speed_4]", "[oral_1][laugh_0][break_2]", 0.22, 0.64, 16, "Deep"
        return "[speed_4]", "[oral_2][laugh_0][break_3]", 0.25, 0.66, 18, "Grounded"
    if any(keyword in normalized_text for keyword in ("soft", "gentle", "sweet", "airy", "breathy", "柔", "甜", "轻", "气声")):
        if gender == "male":
            return "[speed_4]", "[oral_1][laugh_0][break_4]", 0.24, 0.66, 18, "Soft"
        return "[speed_5]", "[oral_1][laugh_0][break_5]", 0.28, 0.68, 20, "Soft"
    if any(keyword in normalized_text for keyword in ("bright", "clear", "energetic", "idol", "亮", "清", "元气", "偶像")):
        if gender == "male":
            return "[speed_5]", "[oral_1][laugh_0][break_3]", 0.25, 0.67, 18, "Clear"
        return "[oral_4][laugh_1][break_3][speed_6]", "[oral_2][laugh_0][break_4]", 0.34, 0.72, 24, "Bright"
    if gender == "male":
        return "[speed_4]", "[oral_2][laugh_0][break_3]", 0.24, 0.66, 18, "Custom"
    return "[speed_5]", "[oral_1][laugh_0][break_4]", 0.28, 0.68, 20, "Custom"


def _apply_literal_readout_bias(
    preset_id: str,
    prompt: str,
    refine_prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[str, str, float, float, int]:
    """把生成参数收紧成“尽量按输入读”的保守模式。"""

    del refine_prompt
    if preset_id == "female_bright_cheerful":
        return (
            prompt,
            "",
            min(temperature, 0.34),
            min(top_p, 0.82),
            min(top_k, 28),
        )
    if preset_id == "female_sad_youth":
        return (
            prompt,
            "",
            min(temperature, 0.22),
            min(top_p, 0.60),
            min(top_k, 12),
        )
    if preset_id == "female_gentle":
        return (
            prompt,
            "",
            min(temperature, 0.20),
            min(top_p, 0.58),
            min(top_k, 10),
        )
    if preset_id == "female_elegant_mature":
        return (
            prompt,
            "",
            min(temperature, 0.21),
            min(top_p, 0.60),
            min(top_k, 12),
        )
    if preset_id == "female_natural":
        return (
            prompt,
            "",
            min(temperature, 0.20),
            min(top_p, 0.58),
            min(top_k, 10),
        )
    if preset_id == "female_heartbroken":
        return (
            prompt,
            "",
            min(temperature, 0.15),
            min(top_p, 0.48),
            min(top_k, 8),
        )
    return (
        prompt,
        "",
        min(temperature, 0.16),
        min(top_p, 0.55),
        min(top_k, 8),
    )


def _override_prompt_speed(prompt: str, speech_rate_id: str | None) -> str:
    """允许客户端显式覆盖 ChatTTS 的 speed token。"""

    normalized_rate = (speech_rate_id or "").strip().lower()
    rate_to_prompt = {
        "slow": "[speed_3]",
        "medium": "[speed_4]",
        "fast": "[speed_8]",
    }
    target_speed_token = rate_to_prompt.get(normalized_rate)
    if not target_speed_token:
        return prompt
    if re.search(r"\[speed_\d+\]", prompt):
        return re.sub(r"\[speed_\d+\]", target_speed_token, prompt)
    return f"{prompt}{target_speed_token}"


def _replace_or_append_token(prompt: str, token_name: str, token_value: int) -> str:
    """按 token 名称替换数值，避免覆盖掉同一条 prompt 里的其它风格 token。"""

    token_pattern = rf"\[{token_name}_\d+\]"
    replacement = f"[{token_name}_{token_value}]"
    if re.search(token_pattern, prompt):
        return re.sub(token_pattern, replacement, prompt)
    return f"{prompt}{replacement}"


def _apply_rate_style_bias(
    preset_id: str,
    speech_rate_id: str | None,
    prompt: str,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[str, float, float, int]:
    """对特定预设的慢档/快档做风格化偏置，而不是只有裸 speed 覆盖。"""

    normalized_rate = (speech_rate_id or "").strip().lower()

    if preset_id == "female_sad_youth" and normalized_rate == "slow":
        prompt = _replace_or_append_token(prompt, "oral", 2)
        prompt = _replace_or_append_token(prompt, "break", 4)
        prompt = _replace_or_append_token(prompt, "speed", 3)
        return prompt, min(temperature, 0.21), min(top_p, 0.60), min(top_k, 12)

    return prompt, temperature, top_p, top_k


def resolve_text_voice_profile(
    gender: str | None,
    tone_mode: str | None,
    tone_preset_id: str | None,
    custom_tone_text: str | None,
    speech_rate_id: str | None = None,
) -> TextVoiceProfile:
    """根据 gender / tone mode / tone preset 解析到最终 ChatTTS 配置。"""

    normalized_gender = (gender or "female").strip().lower()
    preset_spec = _resolve_tone_preset_spec(normalized_gender, tone_preset_id)
    base_preset = TEXT_VOICE_PRESETS[preset_spec.base_voice_preset]
    resolved_prompt = preset_spec.prompt
    resolved_refine_prompt = preset_spec.refine_prompt
    resolved_temperature = preset_spec.temperature
    resolved_top_p = preset_spec.top_p
    resolved_top_k = preset_spec.top_k
    resolved_label = preset_spec.label

    if (tone_mode or "").strip().lower() == "custom" and (custom_tone_text or "").strip():
        (
            resolved_prompt,
            resolved_refine_prompt,
            resolved_temperature,
            resolved_top_p,
            resolved_top_k,
            custom_label,
        ) = _resolve_custom_tone_rule(normalized_gender, custom_tone_text or "")
        resolved_label = f"Custom / {custom_label}"

    (
        resolved_prompt,
        resolved_refine_prompt,
        resolved_temperature,
        resolved_top_p,
        resolved_top_k,
    ) = _apply_literal_readout_bias(
        tone_preset_id or preset_spec.base_voice_preset,
        resolved_prompt,
        resolved_refine_prompt,
        resolved_temperature,
        resolved_top_p,
        resolved_top_k,
    )
    resolved_prompt = _override_prompt_speed(resolved_prompt, speech_rate_id)
    (
        resolved_prompt,
        resolved_temperature,
        resolved_top_p,
        resolved_top_k,
    ) = _apply_rate_style_bias(
        tone_preset_id or preset_spec.base_voice_preset,
        speech_rate_id,
        resolved_prompt,
        resolved_temperature,
        resolved_top_p,
        resolved_top_k,
    )

    return TextVoiceProfile(
        prompt=resolved_prompt,
        refine_prompt=resolved_refine_prompt,
        temperature=resolved_temperature,
        top_p=resolved_top_p,
        top_k=resolved_top_k,
        spk_emb=base_preset["spk_emb"],
        resolved_preset_id=preset_spec.base_voice_preset,
        resolved_tone_label=resolved_label,
    )
