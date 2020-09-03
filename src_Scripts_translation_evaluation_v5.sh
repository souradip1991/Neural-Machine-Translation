SCRIPTNAME=`basename $0`
#Arguments
MODELDIR=$1
TESTFILE=$2
REFTRANSFILE=$3

#export ENGINEDIR=
#export SKIPBPE=
#export SUFFIX=
#export GPUID=

BASE_MODEL=$MODELDIR
DATA_PATH=/home/mcm/BiasInMT/data/EN-ES
RES_PATH=/home/mcm/BiasInMT/scripts/NMT/tmp
SUM_RES_FILE1=${RES_PATH}/TRANS.eval_score.out
SUM_RES_FILE2=${RES_PATH}/TRANS.lex_diversity.out

if [ $# -ne 3 ];then
        echo "Invalid number of arguments. Please provide atleast 3 arguments while running this script."
        echo "Args 1 : Model Directory"
        echo "Args 2 : Test translation file name"
        echo "Args 3 : Reference translation file name"
        echo "Script Usage : sh ${SCRIPTNAME} /home/usenname sample_eng.txt ref_sample_eng.txt"
        exit 1
fi

modelfilecount=`ls -1tr ${MODELDIR} 2>/dev/null | wc -l`

if [ ${modelfilecount} -gt 0 ];then
        echo "Total [ ${modelfilecount} ] models found in the directory [ ${MODELDIR} ]."
else
        echo "ERROR! No models found in the model directory [ ${MODELDIR} ]."
        exit 2
fi

rm -f ${SUM_RES_FILE}

#Dimitar: Get the target language
LANG=$( echo $DATA_PATH | rev | cut -d '-' -f 1 | rev | tr '[:upper:]' '[:lower:]' )

#iterating thorugh different models
for model in `ls -1tr ${MODELDIR}`
do
        #Translate the test file
        SUFFIX=$model
        TRANS_OUTFILE=${TESTFILE}.${SUFFIX}.out
        sh 5_translate_simple.sh ${TESTFILE} ${MODELDIR}/$model $SUFFIX
        #if [ -f ${TRANS_OUTFILE} ];then
                #echo "Output file [ ${TRANS_OUTFILE} ] not generated as it is supposed to be.."
                #exit 3
        #fi

        #Compute and store output of bleu, ter, etc.
        python3 score_bleu_ter.py -r ${REFTRANSFILE} -t ${TRANS_OUTFILE} -l ${LANG} > ${RES_PATH}/${model}_eval_score.out
        echo "${model}" >> ${SUM_RES_FILE1}
        cat ${RES_PATH}/${model}_eval_score.out >> ${SUM_RES_FILE1}

        #Compute and store output of lexical diversity
        python3 score_lexical_diversity.py -f ${TRANS_OUTFILE} > ${RES_PATH}/${model}_lex_diversity.out
        echo "${model}" >> ${SUM_RES_FILE2}
        cat ${RES_PATH}/${model}_lex_diversity.out >> ${SUM_RES_FILE2}

done

#Dimitar: We need to compute and store the lexical diversity for the reference file too.
#Dimitar: this we need to do only for the lexical diversity (no need for BLEU and TER).
python3 score_lexical_diversity.py -f ${REFTRANSFILE} > ${RES_PATH}/tmp_${LANG}_REF_lex_diversity.out
echo "Reference" >> ${SUM_RES_FILE2}
cat ${RES_PATH}/tmp_${LANG}_REF_lex_diversity.out >> ${SUM_RES_FILE2}

