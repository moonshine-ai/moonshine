package ai.moonshine.voice;

public class TranscriberOption {
    public String name;
    public String value;

    public TranscriberOption(String name, String value) {
        this.name = name;
        this.value = value;
    }

    public String name() {
        return name;
    }

    public String value() {
        return value;
    }
}
