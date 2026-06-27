fix_kaldi_dir() {
    local DDIR=$1
    echo "Fixing $DDIR"

    # Backup
    cp $DDIR/utt2spk $DDIR/utt2spk.bak
    cp $DDIR/spk2utt $DDIR/spk2utt.bak

    # Rebuild utt2spk: speaker = prefix of utt_id before "_<rec>_"
    awk '{
        utt = $1; rec = $2
        n = index(utt, "_" rec "_")
        if (n > 0) {
            spk = substr(utt, 1, n-1)
            print utt, spk
        } else {
            print "WARN: cannot parse " utt " (rec=" rec ")" > "/dev/stderr"
        }
    }' $DDIR/segments | sort > $DDIR/utt2spk

    # Rebuild spk2utt from the new utt2spk
    awk '{print $2, $1}' $DDIR/utt2spk | sort | awk '
        BEGIN { prev = "" }
        {
            if ($1 != prev) { if (prev != "") printf "\n"; printf "%s", $1; prev = $1 }
            printf " %s", $2
        }
        END { printf "\n" }
    ' > $DDIR/spk2utt

    echo "  unique speakers: $(awk '{print $2}' $DDIR/utt2spk | sort -u | wc -l)"
    echo "  utt2spk lines:   $(wc -l < $DDIR/utt2spk)"
    echo "  spk2utt lines:   $(wc -l < $DDIR/spk2utt)"
    echo "  sample lines:"
    head -3 $DDIR/utt2spk | sed 's/^/    /'
}

fix_kaldi_dir /root/workspace/der_scoring_RAMC/train_dataset_kaldi
fix_kaldi_dir /root/workspace/der_scoring_RAMC/dev_dataset_kaldi