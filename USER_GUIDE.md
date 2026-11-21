# Quarkon user guide (for people who do not code)

This guide is for colleagues who need the result, not the pipeline. You will not
write code or configuration. The program asks four questions and explains every
problem in plain language.

## 1. Install once

**Windows**

1. Install Python 3.10 or newer from <https://www.python.org/downloads/> and tick
   *Add Python to PATH*.
2. Right-click `install.ps1` -> *Run with PowerShell*, or run:
   `powershell -ExecutionPolicy Bypass -File install.ps1`
3. A **Quarkon** shortcut appears on the desktop.

**Linux / macOS**

```bash
sh install.sh
```

Then start Quarkon any time with the desktop shortcut, or run:

```bash
quarkon launcher
```

The interface opens in your browser at `127.0.0.1`. It is served from your own
computer and is not reachable from the network.

## 2. Create your first project (about two minutes)

1. **Choose the task.** Pick a ready-made task on the left (support triage,
   invoices, contracts, incidents, code review, meeting notes), or describe your
   own task in one or two sentences.
2. **Choose the language** of the examples. The interface language is separate
   and can be changed in the top-right corner (21 languages).
3. **Set the size.** 300 examples is a good start. The card below shows an
   estimated cost before anything is spent.
4. **Check this computer.** Every problem is listed with *what it means* and
   *how to fix it*.

Press **Create project**. You now have a folder with a configuration and example
requests. Nothing has been sent anywhere yet.

## 3. Run the three steps

| Button | What happens | What you need |
|---|---|---|
| **Generate training data** | The teacher model answers your example requests; answers are validated and stored. Resumable: a crash does not lose finished work. | Internet connection, teacher API key |
| **Train the model** | A small model learns from that data on your GPU. | NVIDIA GPU with 6 GB or more |
| **Package for your servers** | The model is converted to GGUF and registered in Ollama. | Ollama installed |

Progress appears in the log. You can close the browser; the step keeps running
until it finishes. To stop, press **Cancel** and re-run later — finished work is
kept.

## 4. Results and review

- **Examples** - how many training samples are ready.
- **Adapter** - whether the trained model exists.
- **Packaged models** - how many GGUF files are ready for deployment.
- **Audit pack** - download JSON or CSV with SHA-256 hashes of every artefact,
  the configuration fingerprint, privacy statistics and the compliance notes your
  security team asks for.

## Problems and what to do

| What you see | What it means | What to do |
|---|---|---|
| *The teacher model key is missing* | No `DEEPSEEK_API_KEY` on this computer | Ask IT for the key; set the environment variable; restart Quarkon |
| *The graphics card has too little memory* | Training needs more GPU memory | Train on a bigger NVIDIA GPU, or lower `min_free_vram_gb`, `batch_size`, `max_seq_length` |
| *No NVIDIA GPU detected* | This computer cannot train | Generate data here, train on a GPU machine |
| *Too few training examples planned* | The model would be fragile | Use 300-500 examples |
| *A secret was found in your input* | An example looks like a password or key | Remove it from the examples and check again |
| *The teacher model could not be reached* | Network, quota or rate limit | Check the connection and key balance; retry — progress is saved |
| *Training stopped* | Out of GPU memory or missing package | Lower batch size, close other GPU apps, or use a larger machine |

Full diagnosis from a terminal: `quarkon doctor`. It prints one line per problem
with the fix.

## Languages

The interface is available in 21 languages today (European languages, Arabic,
Hebrew, Hindi, Chinese, Japanese, Korean). Russian is intentionally not included.
Change the language in the top-right corner, or run:

```bash
quarkon languages            # list what is available
quarkon languages --set de   # switch to German
```

More languages are listed as planned in the same command; each one is a
translation file, not a code change.

## Privacy

- **No telemetry.** Nothing is sent anywhere except the teacher API you configure.
- **Clean room.** Names, e-mails, phones, addresses, card numbers, IBANs, IPs and
  pasted secrets are replaced by realistic surrogates before any request leaves
  the computer. The mapping is kept in memory only and destroyed afterwards.
- **Local first.** Datasets, adapters, model files, logs and reports stay in the
  project folder.
- **Credentials.** The API key is read from an environment variable and is never
  written to configuration, logs or datasets.

## FAQ

**Do I need to know programming?** No. The wizard, the doctor and the log are
written for non-technical users. Terminals are only needed for the two commands
in this guide.

**Can I run it on my laptop?** Generating data, yes. Training needs an NVIDIA GPU
with 6 GB or more.

**Can I stop and continue later?** Yes. Every step is resumable; press the same
button again.

**Where are the files?** Windows: `%USERPROFILE%\Quarkon`. Linux/macOS:
`~/Quarkon`. The interface shows the exact folder after you create a project.

**Is my data used to train someone else's model?** No. Requests go to the teacher
provider you configure; the distilled model is created on your machine.
